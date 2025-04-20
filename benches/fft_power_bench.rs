use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, black_box};
use fft_wasm_lib::FFT;
use num_complex::Complex;

fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_vs_power");
    let size = 512;
    let fft = FFT::new(size);
    // ダミーの windowed_data（1秒ぶんでもOK）
    let windowed_data = vec![1.0_f32; size];

    group.bench_with_input(BenchmarkId::new("FFT only", size), &size, |b, &_s| {
        b.iter(|| {
            // FFT 部分だけ
            let _spec: Vec<Complex<f32>> = fft.fft_real(black_box(&windowed_data));
        })
    });
    group.finish();
}

fn bench_power(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_vs_power");
    let size = 512;
    let fft = FFT::new(size);
    let windowed_data = vec![1.0_f32; size];
    // あらかじめ１周分のスペクトルを作っておく
    let spec: Vec<Complex<f32>> = fft.fft_real(&windowed_data);

    group.bench_with_input(BenchmarkId::new("Power only", size), &size, |b, &_s| {
        b.iter(|| {
            // パワースペクトル＋dB変換だけ
            let mut acc = 0.0;
            for c in black_box(&spec) {
                let sq = c.re * c.re + c.im * c.im;
                acc += 5.0 * sq.log10();
            }
            // acc をどこかに使うことで最適化を防ぐ
            black_box(acc)
        })
    });
    group.finish();
}

fn bench_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_vs_power");
    let size = 512;
    let fft = FFT::new(size);
    let data = vec![1.0_f32; 44100 * 5];
    let window = vec![1.0_f32; 128];

    group.bench_with_input(BenchmarkId::new("Full calc_spectrogram", size), &size, |b, &_s| {
        b.iter(|| {
            // ウィンドウ掛け～FFT～パワー計算をまとめて
            fft_wasm_lib::calc_spectrogram(
                black_box(size),
                black_box(&data),
                black_box(&window),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_fft, bench_power, bench_full);
criterion_main!(benches);
