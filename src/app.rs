use std::io::Read;

use colorous::VIRIDIS;
use egui_plot::{Line, Plot, Points, VLine};
use ndarray::prelude::*;
use ndarray_npy::NpzReader;

use egui::{Color32, Id};
use num_complex::ComplexFloat;
use std::collections::HashMap;

use crate::{PowerSweepConfig, PowerSweepValues};

pub struct ClickThrough {
    gamma: f64,
    psweep: PowerSweepConfig,
    resonator: usize,
    settings: HashMap<usize, BiasSetting>,
    values: PowerSweepValues,
    freq_range: (f64, f64),
    freq_max: (f64, f64),
    atten_range: (f64, f64),
    atten_max: (f64, f64),
    show_mag: bool,
}

impl ClickThrough {
    /// Called once before the first frame.
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        // if let Some(storage) = cc.storage {
        //     return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        // }
        let mut json = String::new();
        std::fs::File::open("./psweepconfig.json")
            .unwrap()
            .read_to_string(&mut json)
            .unwrap();
        let mut pv = NpzReader::new(std::fs::File::open("./psweep.npz").unwrap()).unwrap();
        let psweep: PowerSweepConfig = serde_json::from_str(&json).unwrap();
        let maxo: f64 = psweep.attens.iter().fold(f64::MIN, |a, b| a.max(b.0));
        let mino: f64 = psweep.attens.iter().fold(f64::MAX, |a, b| a.min(b.0));
        let maxf: f64 = psweep
            .sweep_config
            .steps
            .iter()
            .fold(f64::MIN, |a, b| a.max(*b));
        let minf: f64 = psweep
            .sweep_config
            .steps
            .iter()
            .fold(f64::MAX, |a, b| a.min(*b));

        ClickThrough {
            gamma: 1.0,
            resonator: 0,
            psweep,
            values: PowerSweepValues::from_reader(&mut pv),
            settings: HashMap::new(),
            freq_range: (minf, maxf),
            freq_max: (minf, maxf),
            atten_range: (mino, maxo),
            atten_max: (mino, maxo),
            show_mag: true,
        }
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
struct BiasPoint {
    output_atten: usize,
    freq: usize,
}

#[derive(Debug)]
struct BiasSetting {
    output_atten: f64,
    freq: f64,
}

impl eframe::App for ClickThrough {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, _storage: &mut dyn eframe::Storage) {
        // eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::menu::bar(ui, |ui| {
                // NOTE: no File->Quit on web pages!
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }

                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().slider_width = ui.available_width() / 3.;
                ui.add(
                    egui::Slider::new(&mut self.freq_range.0, self.freq_max.0..=self.freq_range.1)
                        .clamping(egui::SliderClamping::Always),
                );
                ui.add(
                    egui::Slider::new(&mut self.freq_range.1, self.freq_range.0..=self.freq_max.1)
                        .clamping(egui::SliderClamping::Always),
                );
                ui.label("Frequency Range");
            });
            ui.horizontal(|ui| {
                ui.spacing_mut().slider_width = ui.available_width() / 3.;
                ui.add(
                    egui::Slider::new(
                        &mut self.atten_range.0,
                        self.atten_max.0..=self.atten_range.1,
                    )
                    .clamping(egui::SliderClamping::Always),
                );
                ui.add(
                    egui::Slider::new(
                        &mut self.atten_range.1,
                        self.atten_range.0..=self.atten_max.1,
                    )
                    .clamping(egui::SliderClamping::Always),
                );
                ui.label("Attenuation Range");
            });
            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut self.resonator, 0..=self.values.iq[0].1.shape()[0] - 1)
                        .clamping(egui::SliderClamping::Always)
                        .text("Resonator"),
                );
                ui.add(egui::Separator::default());
                ui.add(egui::Slider::new(&mut self.gamma, 0.0..=3.0).text("Gamma"));
            });

            let h = ui.available_height();
            ui.horizontal(|ui| {
                ui.set_height(h);
                let iqs: Vec<(usize, f64, Vec<[f64; 2]>, Vec<[f64; 2]>)> = self
                    .values
                    .iq
                    .iter()
                    .enumerate()
                    .filter(|(_, ((o, _), _))| *o >= self.atten_range.0 && *o <= self.atten_range.1)
                    .map(|(ai, ((o, i), iq))| {
                        let gain = (10f64.powf((*i + *o * self.gamma) / 10.)).sqrt();
                        (
                            ai,
                            *o,
                            iq.slice(s![self.resonator, ..])
                                .iter()
                                .enumerate()
                                .filter(|(f, _)| {
                                    let f = self.psweep.sweep_config.steps[*f];
                                    f >= self.freq_range.0 && f <= self.freq_range.1
                                })
                                .map(|(_, v)| [v.re as f64 * gain, v.im as f64 * gain])
                                .collect(),
                            iq.slice(s![self.resonator, ..])
                                .iter()
                                .enumerate()
                                .filter(|(f, _)| {
                                    let f = self.psweep.sweep_config.steps[*f];
                                    f >= self.freq_range.0 && f <= self.freq_range.1
                                })
                                .map(|(f, c)| {
                                    [self.psweep.sweep_config.steps[f], gain * c.abs() as f64]
                                })
                                .collect(),
                        )
                    })
                    .collect();

                let fmap: Vec<usize> = self
                    .psweep
                    .sweep_config
                    .steps
                    .iter()
                    .enumerate()
                    .filter(|(fi, f)| **f >= self.freq_range.0 && **f <= self.freq_range.1)
                    .map(|(fi, _)| fi)
                    .collect();

                let mut ids: HashMap<Id, BiasPoint> =
                    HashMap::with_capacity(self.values.iq.len() * 1024);

                let pr = Plot::new(format!("Clickey{}{}", self.resonator, self.gamma))
                    .show_axes([false, false])
                    .width(ui.available_width() / 2.)
                    .data_aspect(1.0)
                    .auto_bounds([true, true].into())
                    .show(ui, |plotui| {
                        iqs.iter().for_each(|(ai, o, l, _)| {
                            let t = (*o - self.atten_range.0)
                                / (self.atten_range.1 - self.atten_range.0);
                            let t = 1. - t;
                            let color = VIRIDIS.eval_continuous(t);
                            let color = Color32::from_rgb(color.r, color.g, color.b);
                            for (num, point) in l.iter().enumerate() {
                                let bp = BiasPoint {
                                    output_atten: *ai,
                                    freq: fmap[num],
                                };
                                let id = Id::new(bp);
                                ids.insert(id, bp);
                                plotui.points(
                                    Points::new(vec![*point]).color(color).radius(4.).id(id),
                                )
                            }
                            plotui.line(Line::new(l.clone()).color(color).allow_hover(false))
                        })
                    });

                let mut bp = None;
                if let Some(h) = pr.hovered_plot_item {
                    bp = ids.get(&h);
                }

                Plot::new("Showey")
                    .show_axes([true, true])
                    .width(ui.available_width())
                    .allow_zoom(false)
                    .allow_scroll(false)
                    .allow_drag(false)
                    .show(ui, |plotui| {
                        if self.show_mag {
                            iqs.iter().for_each(|(_, o, _, l)| {
                                let t = (*o - self.atten_range.0)
                                    / (self.atten_range.1 - self.atten_range.0);
                                let t = 1. - t;
                                let color = VIRIDIS.eval_continuous(t);
                                let mut color = Color32::from_rgb(color.r, color.g, color.b);
                                if bp.is_some() {
                                    color = color.gamma_multiply(0.1);
                                }
                                plotui.line(Line::new(l.clone()).color(color).allow_hover(false))
                            });
                        }
                        if let Some(bp) = bp {
                            plotui.vline(VLine::new(self.psweep.sweep_config.steps[bp.freq]));
                            for (ai, _, _, v) in iqs.iter() {
                                if *ai == bp.output_atten {
                                    plotui.line(Line::new(v.clone()).highlight(self.show_mag));
                                    break;
                                }
                            }
                            plotui.set_auto_bounds([true, true].into());
                        }
                    });

                if pr.response.clicked() {
                    if let Some(bp) = bp {
                        let bs = BiasSetting {
                            output_atten: self.psweep.attens[bp.output_atten].0,
                            freq: self.psweep.sweep_config.steps[bp.freq],
                        };
                        self.settings.insert(self.resonator, bs);
                        self.resonator += 1;
                    }
                }
            });

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                egui::warn_if_debug_build(ui);
            });
        });
    }
}
