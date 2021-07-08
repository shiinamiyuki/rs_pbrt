// std
use std::borrow::Borrow;
use std::cell::Cell;
use std::f32::consts::PI;
use std::iter::once;
use std::sync::{Arc, RwLock};

mod nn_v2;
use crate::blockqueue::BlockQueue;
// pbrt
// use crate::core::bssrdf::Bssrdf;
use crate::core::camera::{Camera, CameraSample};
use crate::core::geometry::{
    pnt2_inside_exclusivei, pnt3_inside_bnd3, spherical_phi, spherical_theta, vec3_abs_dot_nrmf,
    vec3_dot_nrmf, Bounds3f, Point2i, Point3f, Vector2f,
};
use crate::core::geometry::{Bounds2i, Point2f, Ray, Vector2i, Vector3f};
use crate::core::integrator::uniform_sample_one_light;
use crate::core::interaction::{Interaction, SurfaceInteraction};
use crate::core::lightdistrib::create_light_sample_distribution;
use crate::core::lightdistrib::LightDistribution;
use crate::core::material::TransportMode;
use crate::core::pbrt::{Float, Spectrum};
use crate::core::reflection::BxdfType;
use crate::core::sampler::Sampler;
use crate::core::sampling::Distribution1D;
use crate::core::scene::Scene;

use nn_v2::*;

use nalgebra as na;
use nalgebra_glm as glm;
use rand::Rng;

const POSITION_INPUTS: usize = 3;
const POSITION_FREQ: usize = 12;
const NOMRAL_INPUTS: usize = 2;
const DIR_INPUTS: usize = 2;
const ALBEDO_INPUTS: usize = 3;
const ROUGHNESS_INPUTS: usize = 1;
const METALLIC_INPUTS: usize = 1;
const ONE_BLOB_SIZE: usize = 4;
const INPUT_SIZE: usize = POSITION_INPUTS
    + NOMRAL_INPUTS
    + DIR_INPUTS
    + ALBEDO_INPUTS
    + METALLIC_INPUTS
    + ROUGHNESS_INPUTS;
const MAPPED_SIZE: usize = POSITION_INPUTS * POSITION_FREQ
    + ONE_BLOB_SIZE * (NOMRAL_INPUTS + DIR_INPUTS)
    + ONE_BLOB_SIZE * ROUGHNESS_INPUTS
    + METALLIC_INPUTS
    + ALBEDO_INPUTS;
const ALBEDO_OFFSET: usize =
    POSITION_INPUTS + NOMRAL_INPUTS + DIR_INPUTS + ROUGHNESS_INPUTS + METALLIC_INPUTS;
// type FeatureMat = na::SMatrix<f32, { FEATURE_SIZE }, 5>;
type InputVec = na::SVector<f32, { INPUT_SIZE }>;

fn one_blob_encoding(input: f32, i: usize) -> f32 {
    assert!(0.0 <= input && input <= 1.0);
    let sigma = 1.0 / ONE_BLOB_SIZE as f32;
    let x = (i as f32 / ONE_BLOB_SIZE as f32 - input) as f32;
    1.0 / ((2.0 * std::f32::consts::PI).sqrt() * sigma) * (-x * x / (2.0 * sigma * sigma)).exp()
    // input
}

fn nrc_encoding(v: &na::DMatrix<f32>) -> na::DMatrix<f32> {
    let mut u: na::DMatrix<f32> = na::DMatrix::zeros(MAPPED_SIZE, v.ncols());
    assert!(v.nrows() == INPUT_SIZE);
    for c in 0..v.ncols() {
        for i in 0..POSITION_INPUTS {
            for f in 0..POSITION_FREQ {
                if f > 0 {
                    u[(i * POSITION_FREQ + f, c)] =
                        (v[(i, c)] * 2.0f32.powi((f - 1) as i32) * PI).sin();
                } else {
                    u[(i * POSITION_FREQ + f, c)] = v[(i, c)];
                }
            }
        }
        let offset_u = POSITION_INPUTS * POSITION_FREQ;
        let offset_v = POSITION_INPUTS;
        for i in 0..NOMRAL_INPUTS {
            for k in 0..ONE_BLOB_SIZE {
                u[(offset_u + i * ONE_BLOB_SIZE + k, c)] =
                    one_blob_encoding(v[(offset_v + i, c)], k);
            }
        }
        let offset_u = offset_u + NOMRAL_INPUTS * ONE_BLOB_SIZE;
        let offset_v = offset_v + NOMRAL_INPUTS;
        for i in 0..DIR_INPUTS {
            for k in 0..ONE_BLOB_SIZE {
                u[(offset_u + i * ONE_BLOB_SIZE + k, c)] =
                    one_blob_encoding(v[(offset_v + i, c)], k);
            }
        }
        let offset_u = offset_u + DIR_INPUTS * ONE_BLOB_SIZE;
        let offset_v = offset_v + DIR_INPUTS;
        for k in 0..ONE_BLOB_SIZE {
            u[(offset_u + k, c)] = one_blob_encoding(1.0 - (-v[(offset_v, c)]).exp(), k);
        }
        let offset_u = offset_u + ROUGHNESS_INPUTS * ONE_BLOB_SIZE;
        let offset_v = offset_v + ROUGHNESS_INPUTS;
        for k in 0..(METALLIC_INPUTS + ALBEDO_INPUTS) {
            u[(offset_u + k, c)] = v[(offset_v + k, c)];
        }
        assert!(offset_u + METALLIC_INPUTS + ALBEDO_INPUTS == MAPPED_SIZE);
        assert!(offset_v + METALLIC_INPUTS + ALBEDO_INPUTS == INPUT_SIZE);
    }
    assert!(u.nrows() == MAPPED_SIZE);
    u
}
// position_encoding_func_v3!(nrc_encoding, INPUT_SIZE, FEATURE_SIZE, MAX_FREQ);

fn sph(v: &Vector3f) -> Vector2f {
    let phi = spherical_phi(v);
    let theta = spherical_theta(v);
    // vec2(v.x / PI, v.y / (2.0 * PI))
    Vector2f {
        x: theta / PI,
        y: theta / (2.0 * PI),
    }
}

fn tone_mapping(x: f32) -> f32 {
    x / (x + 1.0)
}
fn inv_tone_mapping(x: f32) -> f32 {
    x / (1.0 - x)
}
struct TrainRecord {
    queue: Vec<f32>,
    target: Vec<f32>,
}
struct RadianceCache {
    model: Arc<MLP>,
    bound: Bounds3f,
    query_queue: RwLock<Vec<f32>>,
    query_result: RwLock<na::DMatrix<f32>>,
    train: RwLock<TrainRecord>,
}
impl Clone for RadianceCache {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            bound: self.bound,
            query_result: RwLock::new(na::DMatrix::zeros(0, 0)),
            query_queue: RwLock::new(vec![]),
            train: RwLock::new({
                TrainRecord {
                    target: vec![],
                    queue: vec![],
                }
            }),
        }
    }
}
#[derive(Clone, Copy)]
struct BsdfInfo {
    albedo: Spectrum,
    metallic: Float,
    roughness: Float,
}
#[derive(Clone, Copy)]
struct QueryRecord {
    n: Vector2f,
    info: BsdfInfo,
    x: Vector3f,
    dir: Vector2f,
}

#[derive(Clone, Copy)]
struct PathState {
    beta: Spectrum,
    li: Spectrum,
    query_index: Option<usize>,
    thread_index: usize,
}
impl RadianceCache {
    fn new(model: MLP) -> Self {
        Self {
            model: Arc::new(model),
            bound: Bounds3f {
                p_min: Point3f {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0,
                } * -500.0,
                p_max: Point3f {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0,
                } * 500.0,
            },
            query_queue: RwLock::new(vec![]),
            query_result: RwLock::new(na::DMatrix::zeros(0, 0)),
            train: RwLock::new(TrainRecord {
                queue: vec![],
                target: vec![],
            }),
        }
    }
    fn get_input_vec(r: &QueryRecord) -> InputVec {
        InputVec::from_iterator(
            [r.x.x, r.x.y, r.x.z]
                .iter()
                .map(|x| *x)
                .chain([r.n.x, r.n.y].iter().map(|x| *x))
                .chain([r.dir.x, r.dir.y].iter().map(|x| *x))
                .chain(once(r.info.roughness))
                .chain(once(r.info.metallic))
                .chain(r.info.albedo.c.iter().map(|x| *x as f32)),
        )
    }
    // returns index to query_result
    fn record_infer(&self, r: &QueryRecord) -> Option<usize> {
        if !pnt3_inside_bnd3(
            &Point3f {
                x: r.x.x,
                y: r.x.y,
                z: r.x.z,
            },
            &self.bound,
        ) {
            return None;
        }
        let v = Self::get_input_vec(r);
        let mut queue = self.query_queue.write().unwrap();
        let idx = queue.len() / INPUT_SIZE;
        queue.extend(v.iter());
        // let s: Spectrum = self.model.infer(&nrc_encoding(&v)).into();
        Some(idx)
    }
    fn infer_single(&self, r: &QueryRecord) -> Option<Spectrum> {
        if !pnt3_inside_bnd3(
            &Point3f {
                x: r.x.x,
                y: r.x.y,
                z: r.x.z,
            },
            &self.bound,
        ) {
            return None;
        }
        let v = Self::get_input_vec(r);
        let mut r = self
            .model
            .infer(nrc_encoding(&na::DMatrix::from_column_slice(
                v.nrows(),
                1,
                v.as_slice(),
            )));
        r.iter_mut().for_each(|x| *x = inv_tone_mapping(*x));
        assert!(r.nrows() == 3 && r.ncols() == 1);
        Some(Spectrum {
            // samples: na::SVector::from_iterator(r.iter().map(|x| *x as Float)),
            c: [r[0], r[1], r[2]],
        })
    }
    fn record_train(&self, r: &QueryRecord, target: &Spectrum) {
        if !pnt3_inside_bnd3(
            &Point3f {
                x: r.x.x,
                y: r.x.y,
                z: r.x.z,
            },
            &self.bound,
        ) {
            return;
        }
        let v = Self::get_input_vec(r);
        let mut train = self.train.write().unwrap();
        assert_eq!(
            train.queue.len() / INPUT_SIZE,
            train.target.len() / 3,
            "train record lens diff! {} and {}",
            train.queue.len() / INPUT_SIZE,
            train.target.len() / 3
        );
        train.queue.extend(v.iter());
        let target = target
            .c
            .iter()
            .zip(r.info.albedo.c.iter())
            .map(|(a, b)| if *b != 0.0 { *a / *b } else { 0.0 });
        train.target.extend(target.map(|x| tone_mapping(x as f32)));
    }
    fn infer(&self) {
        {
            let queue = self.query_queue.write().unwrap();
            assert!(queue.len() % INPUT_SIZE == 0);
            let mut inputs = na::DMatrix::<f32>::from_column_slice(
                INPUT_SIZE,
                queue.len() / INPUT_SIZE,
                &queue[..],
            );
            inputs = nrc_encoding(&inputs);
            let mut result = self.query_result.write().unwrap();
            *result = {
                let mut r = self.model.infer(inputs);
                r.iter_mut().for_each(|x| *x = inv_tone_mapping(*x));
                for i in 0..r.ncols() {
                    for j in 0..r.nrows() {
                        r[(j, i)] *= queue[i * INPUT_SIZE + ALBEDO_OFFSET + j];
                    }
                }
                r
            };
        }
        {
            let mut queue = self.query_queue.write().unwrap();
            queue.clear();
        }
    }
    fn train(&mut self) -> f32 {
        let mut train = self.train.write().unwrap();
        assert!(train.queue.len() % INPUT_SIZE == 0);
        assert!(train.target.len() % 3 == 0);
        let queue = &train.queue;
        let mut inputs =
            na::DMatrix::<f32>::from_column_slice(INPUT_SIZE, queue.len() / INPUT_SIZE, &queue[..]);
        inputs = nrc_encoding(&inputs);
        let targets =
            na::DMatrix::<f32>::from_column_slice(3, &train.target.len() / 3, &train.target[..]);
        let loss = Arc::get_mut(&mut self.model)
            .unwrap()
            .train(inputs, &targets, Loss::RelativeL2);
        train.queue.clear();
        train.target.clear();
        loss
    }
}

// see path.h

/// Path Tracing (Global Illumination) - uses the render loop of a
/// [SamplerIntegrator](../../core/integrator/enum.SamplerIntegrator.html)
pub struct NRCPathIntegrator {
    // inherited from SamplerIntegrator (see integrator.h)
    pub camera: Arc<Camera>,
    pub sampler: Box<Sampler>,
    pixel_bounds: Bounds2i,
    // see path.h
    max_depth: u32,
    rr_threshold: Float,           // 1.0
    light_sample_strategy: String, // "spatial"
    light_distribution: Option<Arc<LightDistribution>>,
}

impl NRCPathIntegrator {
    pub fn new(
        max_depth: u32,
        camera: Arc<Camera>,
        sampler: Box<Sampler>,
        pixel_bounds: Bounds2i,
        rr_threshold: Float,
        light_sample_strategy: String,
    ) -> Self {
        Self {
            camera,
            sampler,
            pixel_bounds,
            max_depth,
            rr_threshold,
            light_sample_strategy,
            light_distribution: None,
        }
    }
    pub fn preprocess(&mut self, scene: &Scene) {
        self.light_distribution =
            create_light_sample_distribution(self.light_sample_strategy.clone(), scene);
    }
    pub fn li(
        &self,
        r: &Ray,
        scene: &Scene,
        sampler: &mut Sampler,
        // arena: &mut Arena,
        _depth: i32,
    ) -> Spectrum {
        // TODO: ProfilePhase p(Prof::SamplerIntegratorLi);
        let mut l: Spectrum = Spectrum::default();
        let mut beta: Spectrum = Spectrum::new(1.0 as Float);
        let mut ray: Ray = Ray {
            o: r.o,
            d: r.d,
            t_max: Cell::new(r.t_max.get()),
            time: r.time,
            differential: r.differential,
            medium: r.medium.clone(),
        };
        let mut specular_bounce: bool = false;
        let mut bounces: u32 = 0_u32;
        // Added after book publication: etaScale tracks the
        // accumulated effect of radiance scaling due to rays passing
        // through refractive boundaries (see the derivation on p. 527
        // of the third edition). We track this value in order to
        // remove it from beta when we apply Russian roulette; this is
        // worthwhile, since it lets us sometimes avoid terminating
        // refracted rays that are about to be refracted back out of a
        // medium and thus have their beta value increased.
        let mut eta_scale: Float = 1.0;
        loop {
            // find next path vertex and accumulate contribution
            // println!("Path tracer bounce {:?}, current L = {:?}, beta = {:?}",
            //          bounces, l, beta);
            // intersect _ray_ with scene and store intersection in _isect_
            let mut isect: SurfaceInteraction = SurfaceInteraction::default();
            if scene.intersect(&ray, &mut isect) {
                // possibly add emitted light at intersection
                if bounces == 0 || specular_bounce {
                    // add emitted light at path vertex
                    l += beta * isect.le(&-ray.d);
                    // println!("Added Le -> L = {:?}", l);
                }
                // terminate path if _maxDepth_ was reached
                if bounces >= self.max_depth {
                    break;
                }
                // compute scattering functions and skip over medium boundaries
                let mode: TransportMode = TransportMode::Radiance;
                isect.compute_scattering_functions(&ray, true, mode);
                if let Some(ref _bsdf) = isect.bsdf {
                    // we are fine (for below)
                } else {
                    // TODO: println!("Skipping intersection due to null bsdf");
                    ray = isect.spawn_ray(&ray.d);
                    // bounces--;
                    continue;
                }
                if let Some(ref light_distribution) = self.light_distribution {
                    let distrib: Arc<Distribution1D> = light_distribution.lookup(&isect.common.p);
                    // Sample illumination from lights to find path contribution.
                    // (But skip this for perfectly specular BSDFs.)
                    let bsdf_flags: u8 = BxdfType::BsdfAll as u8 & !(BxdfType::BsdfSpecular as u8);
                    if let Some(ref bsdf) = isect.bsdf {
                        if bsdf.num_components(bsdf_flags) > 0 {
                            // TODO: ++total_paths;
                            let it: &SurfaceInteraction = isect.borrow();
                            let ld: Spectrum = beta
                                * uniform_sample_one_light(
                                    it,
                                    scene,
                                    sampler,
                                    false,
                                    Some(&distrib),
                                );
                            // TODO: println!("Sampled direct lighting Ld = {:?}", ld);
                            // TODO: if ld.is_black() {
                            //     ++zero_radiance_paths;
                            // }
                            assert!(ld.y() >= 0.0 as Float, "ld = {:?}", ld);
                            l += ld;
                        }
                        // Sample BSDF to get new path direction
                        let wo: Vector3f = -ray.d;
                        let mut wi: Vector3f = Vector3f::default();
                        let mut pdf: Float = 0.0 as Float;
                        let bsdf_flags: u8 = BxdfType::BsdfAll as u8;
                        let mut sampled_type: u8 = u8::max_value(); // != 0
                        let f: Spectrum = bsdf.sample_f(
                            &wo,
                            &mut wi,
                            &sampler.get_2d(),
                            &mut pdf,
                            bsdf_flags,
                            &mut sampled_type,
                        );

                        // println!("Sampled BSDF, f = {:?}, pdf = {:?}", f, pdf);
                        if f.is_black() || pdf == 0.0 as Float {
                            break;
                        }
                        beta *= (f * vec3_abs_dot_nrmf(&wi, &isect.shading.n)) / pdf;
                        // println!("Updated beta = {:?}", beta);
                        assert!(beta.y() >= 0.0 as Float);
                        assert!(
                            !(beta.y().is_infinite()),
                            "[{:#?}, {:?}] = ({:#?} * dot({:#?}, {:#?})) / {:?}",
                            sampler.get_current_pixel(),
                            sampler.get_current_sample_number(),
                            f,
                            wi,
                            isect.shading.n,
                            pdf
                        );
                        specular_bounce = (sampled_type & BxdfType::BsdfSpecular as u8) != 0_u8;
                        if ((sampled_type & BxdfType::BsdfSpecular as u8) != 0_u8)
                            && ((sampled_type & BxdfType::BsdfTransmission as u8) != 0_u8)
                        {
                            let eta: Float = bsdf.eta;
                            // Update the term that tracks radiance
                            // scaling for refraction depending on
                            // whether the ray is entering or leaving
                            // the medium.
                            if vec3_dot_nrmf(&wo, &isect.common.n) > 0.0 as Float {
                                eta_scale *= eta * eta;
                            } else {
                                eta_scale *= 1.0 as Float / (eta * eta);
                            }
                        }
                        ray = isect.spawn_ray(&wi);

                        // account for subsurface scattering, if applicable
                        if let Some(ref bssrdf) = isect.bssrdf {
                            if (sampled_type & BxdfType::BsdfTransmission as u8) != 0_u8 {
                                // importance sample the BSSRDF
                                let s2: Point2f = sampler.get_2d();
                                let s1: Float = sampler.get_1d();
                                let (s, pi_opt) = bssrdf.sample_s(
                                    // the next three (extra) parameters are used for SeparableBssrdfAdapter
                                    bssrdf.clone(),
                                    bssrdf.mode,
                                    bssrdf.eta,
                                    // done
                                    scene,
                                    s1,
                                    s2,
                                    &mut pdf,
                                );
                                if s.is_black() || pdf == 0.0 as Float {
                                    break;
                                }
                                assert!(!(beta.y().is_infinite()));
                                beta *= s / pdf;
                                if let Some(pi) = pi_opt {
                                    // account for the direct subsurface scattering component
                                    let distrib: Arc<Distribution1D> =
                                        light_distribution.lookup(&pi.common.p);
                                    l += beta
                                        * uniform_sample_one_light(
                                            &pi,
                                            scene,
                                            sampler,
                                            false,
                                            Some(&distrib),
                                        );
                                    // account for the indirect subsurface scattering component
                                    let mut wi: Vector3f = Vector3f::default();
                                    let mut pdf: Float = 0.0 as Float;
                                    let bsdf_flags: u8 = BxdfType::BsdfAll as u8;
                                    let mut sampled_type: u8 = u8::max_value(); // != 0
                                    if let Some(ref bsdf) = pi.bsdf {
                                        let f: Spectrum = bsdf.sample_f(
                                            &pi.common.wo,
                                            &mut wi,
                                            &sampler.get_2d(),
                                            &mut pdf,
                                            bsdf_flags,
                                            &mut sampled_type,
                                        );
                                        if f.is_black() || pdf == 0.0 as Float {
                                            break;
                                        }
                                        beta *= f * vec3_abs_dot_nrmf(&wi, &pi.shading.n) / pdf;
                                        assert!(!(beta.y().is_infinite()));
                                        specular_bounce =
                                            (sampled_type & BxdfType::BsdfSpecular as u8) != 0_u8;
                                        ray = pi.spawn_ray(&wi);
                                    }
                                }
                            }
                        }

                        // Possibly terminate the path with Russian roulette.
                        // Factor out radiance scaling due to refraction in rr_beta.
                        let rr_beta: Spectrum = beta * eta_scale;
                        if rr_beta.max_component_value() < self.rr_threshold && bounces > 3 {
                            let q: Float =
                                (0.05 as Float).max(1.0 as Float - rr_beta.max_component_value());
                            if sampler.get_1d() < q {
                                break;
                            }
                            beta /= 1.0 as Float - q;
                            assert!(!(beta.y().is_infinite()));
                        }
                    } else {
                        println!("TODO: if let Some(ref bsdf) = isect.bsdf failed");
                    }
                }
            } else {
                // add emitted light from the environment
                if bounces == 0 || specular_bounce {
                    // for (const auto &light : scene.infiniteLights)
                    for light in &scene.infinite_lights {
                        l += beta * light.le(&ray);
                    }
                    // println!("Added infinite area lights -> L = {:?}", l);
                }
                // terminate path if ray escaped
                break;
            }
            bounces += 1_u32;
        }
        l
    }
    pub fn get_camera(&self) -> Arc<Camera> {
        self.camera.clone()
    }
    pub fn get_sampler(&self) -> &Sampler {
        &self.sampler
    }
    pub fn get_pixel_bounds(&self) -> Bounds2i {
        self.pixel_bounds
    }

    pub fn render(&mut self, scene: &Scene, num_threads: u8) {
        let film = self.get_camera().get_film();
        let sample_bounds: Bounds2i = film.get_sample_bounds();
        self.preprocess(scene);
        let sample_extent: Vector2i = sample_bounds.diagonal();
        let tile_size: i32 = 16;
        let x: i32 = (sample_extent.x + tile_size - 1) / tile_size;
        let y: i32 = (sample_extent.y + tile_size - 1) / tile_size;
        let n_tiles: Point2i = Point2i { x, y };
        // TODO: ProgressReporter reporter(nTiles.x * nTiles.y, "Rendering");
        let num_cores = if num_threads == 0_u8 {
            num_cpus::get()
        } else {
            num_threads as usize
        };
        println!("Rendering with {:?} thread(s) ...", num_cores);
        {
            let block_queue = BlockQueue::new(
                (
                    (n_tiles.x * tile_size) as u32,
                    (n_tiles.y * tile_size) as u32,
                ),
                (tile_size as u32, tile_size as u32),
                (0, 0),
            );
            let integrator = &self;
            let bq = &block_queue;
            let sampler = &self.get_sampler();
            let camera = &self.get_camera();
            let film = &film;
            let pixel_bounds = &self.get_pixel_bounds();
            crossbeam::scope(|scope| {
                let (pixel_tx, pixel_rx) = crossbeam_channel::bounded(num_cores);
                // spawn worker threads
                for _ in 0..num_cores {
                    let pixel_tx = pixel_tx.clone();
                    let mut tile_sampler: Box<Sampler> = sampler.clone_with_seed(0_u64);
                    scope.spawn(move |_| {
                        while let Some((x, y)) = bq.next() {
                            let tile: Point2i = Point2i {
                                x: x as i32,
                                y: y as i32,
                            };
                            let seed: i32 = tile.y * n_tiles.x + tile.x;
                            tile_sampler.reseed(seed as u64);
                            let x0: i32 = sample_bounds.p_min.x + tile.x * tile_size;
                            let x1: i32 = std::cmp::min(x0 + tile_size, sample_bounds.p_max.x);
                            let y0: i32 = sample_bounds.p_min.y + tile.y * tile_size;
                            let y1: i32 = std::cmp::min(y0 + tile_size, sample_bounds.p_max.y);
                            let tile_bounds: Bounds2i =
                                Bounds2i::new(Point2i { x: x0, y: y0 }, Point2i { x: x1, y: y1 });
                            // println!("Starting image tile {:?}", tile_bounds);
                            let mut film_tile = film.get_film_tile(&tile_bounds);
                            for pixel in &tile_bounds {
                                tile_sampler.start_pixel(pixel);
                                if !pnt2_inside_exclusivei(pixel, &pixel_bounds) {
                                    continue;
                                }
                                let mut done: bool = false;
                                while !done {
                                    // let's use the copy_arena crate instead of pbrt's MemoryArena
                                    // let mut arena: Arena = Arena::with_capacity(262144); // 256kB

                                    // initialize _CameraSample_ for current sample
                                    let camera_sample: CameraSample =
                                        tile_sampler.get_camera_sample(pixel);
                                    // generate camera ray for current sample
                                    let mut ray: Ray = Ray::default();
                                    let ray_weight: Float =
                                        camera.generate_ray_differential(&camera_sample, &mut ray);
                                    ray.scale_differentials(
                                        1.0 as Float
                                            / (tile_sampler.get_samples_per_pixel() as Float)
                                                .sqrt(),
                                    );
                                    // TODO: ++nCameraRays;
                                    // evaluate radiance along camera ray
                                    let mut l: Spectrum = Spectrum::new(0.0 as Float);
                                    let y: Float = l.y();
                                    if ray_weight > 0.0 {
                                        // ADDED
                                        let clipping_start: Float = camera.get_clipping_start();
                                        if clipping_start > 0.0 as Float {
                                            // adjust ray origin for near clipping
                                            ray.o = ray.position(clipping_start);
                                        }
                                        // ADDED
                                        l = integrator.li(
                                            &mut ray,
                                            scene,
                                            &mut tile_sampler, // &mut arena,
                                            0_i32,
                                        );
                                    }
                                    if l.has_nans() {
                                        println!(
                                            "Not-a-number radiance value returned for pixel \
                                                     ({:?}, {:?}), sample {:?}. Setting to black.",
                                            pixel.x,
                                            pixel.y,
                                            tile_sampler.get_current_sample_number()
                                        );
                                        l = Spectrum::new(0.0);
                                    } else if y < -10.0e-5 as Float {
                                        println!(
                                            "Negative luminance value, {:?}, returned for pixel \
                                                 ({:?}, {:?}), sample {:?}. Setting to black.",
                                            y,
                                            pixel.x,
                                            pixel.y,
                                            tile_sampler.get_current_sample_number()
                                        );
                                        l = Spectrum::new(0.0);
                                    } else if y.is_infinite() {
                                        println!(
                                            "Infinite luminance value returned for pixel ({:?}, \
                                                 {:?}), sample {:?}. Setting to black.",
                                            pixel.x,
                                            pixel.y,
                                            tile_sampler.get_current_sample_number()
                                        );
                                        l = Spectrum::new(0.0);
                                    }
                                    // println!("Camera sample: {:?} -> ray: {:?} -> L = {:?}",
                                    //          camera_sample, ray, l);
                                    // add camera ray's contribution to image
                                    film_tile.add_sample(camera_sample.p_film, &mut l, ray_weight);
                                    done = !tile_sampler.start_next_sample();
                                } // arena is dropped here !
                            }
                            // send the tile through the channel to main thread
                            pixel_tx
                                .send(film_tile)
                                .unwrap_or_else(|_| panic!("Failed to send tile"));
                        }
                    });
                }
                // spawn thread to collect pixels and render image to file
                scope.spawn(move |_| {
                    for _ in pbr::PbIter::new(0..bq.len()) {
                        let film_tile = pixel_rx.recv().unwrap();
                        // merge image tile into _Film_
                        film.merge_film_tile(&film_tile);
                    }
                });
            })
            .unwrap();
        }
        film.write_image(1.0 as Float);
    }
}
