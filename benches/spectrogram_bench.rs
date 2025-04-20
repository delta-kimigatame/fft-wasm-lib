// benches/spectrogram_compare.rs
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, black_box};
use fft_wasm_lib::{calc_spectrogram, calc_spectrogram_with_rfft};

fn bench_spectrograms(c: &mut Criterion) {
    // ベンチ用の大きなバッファは外で一度だけ確保
    let sample_rate = 44_100;
    let duration_secs = 5;
    let data: Vec<f32> = vec![1.0; sample_rate * duration_secs];
    let window: Vec<f32> = vec![1.0; 128];

    let mut group = c.benchmark_group("spectrogram");
    // いろんな FFT サイズで比較したければここに追加
    for &size in &[512] {
        group.bench_with_input(
            BenchmarkId::new("FFT (complex)", size),
            &size,
            |b, &s| {
                b.iter(|| {
                    // black_box で最適化の邪魔を防ぎつつ呼び出し
                    calc_spectrogram(black_box(s), black_box(&data), black_box(&window))
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("RFFT (real→complex)", size),
            &size,
            |b, &s| {
                b.iter(|| {
                    calc_spectrogram_with_rfft(black_box(s), black_box(&data), black_box(&window))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_spectrograms);
criterion_main!(benches);
