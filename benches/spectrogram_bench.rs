use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fft_wasm_lib::calc_spectrogram;

fn bench_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectrogram");
    let size = 512;
    let data = vec![1.0_f64; 44100*5];// 実際を想定した44100Hz 5秒分のデータ
    let window = vec![1.0_f64; 128];

    group.bench_with_input(
        BenchmarkId::new("all-ones", size),
        &size,
        |b, &_s| b.iter(|| calc_spectrogram(size, &data, &window)),
    );
    group.finish();
}

criterion_group!(benches, bench_spectrogram);
criterion_main!(benches);