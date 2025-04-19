use wasm_bindgen::prelude::*;
use num_complex::Complex;
use std::{ f64::consts::PI };

/// 2進数表現のビットを逆転します。
///
/// # 引数
/// - `k`:ビット長
/// - `n`:入力値
///
/// # 戻り値
/// 逆転後の整数
fn rev_bit(k: usize, n: usize) -> usize {
    let mut r = 0;
    for i in 0..k {
        r = (r << 1) | ((n >> i) & 1);
    }
    r
}

/// FFT ライブラリを表す構造体
///
/// この構造体は、事前に計算した回転因子やビット逆転インデックスを保持し、
/// FFT／iFFT のメソッドを提供します。
pub struct FFT {
    /// FFT のサイズ（2 のべき乗でなければなりません）
    size: usize,

    /// size の 2 ログ（`log2(size)`）
    k: usize,

    /// FFT 用の回転因子（`e^{-2πi * n / size}` の配列）
    twiddle: Vec<Complex<f64>>,

    /// iFFT 用の回転因子（`e^{+2πi * n / size}` の配列）
    itwiddle: Vec<Complex<f64>>,

    /// ビット逆転順序のインデックス（`rev_bit` で得られる並び替え用テーブル）
    rev_indices: Vec<usize>,
}
impl FFT {
    /// コンストラクタ
    ///
    /// # 引数
    /// - `size`:fftのサイズ。2のべき乗である必要がある。
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "size must be power of 2");
        let k = (size as f64).log2() as usize;
        let t = (-2.0 * PI) / (size as f64);
        let it = (2.0 * PI) / (size as f64);
        let mut twiddle = Vec::with_capacity(size);
        let mut itwiddle = Vec::with_capacity(size);
        for i in 0..size {
            twiddle.push(Complex::from_polar(1.0, t * (i as f64)));
            itwiddle.push(Complex::from_polar(1.0, it * (i as f64)));
        }
        let mut rev_indices = vec![0; size];
        for i in 0..size {
            rev_indices[i] = rev_bit(k, i);
        }
        FFT {
            size,
            k,
            twiddle,
            itwiddle,
            rev_indices,
        }
    }

    /// FFTの共通処理部分の、元のデータが実数か複素数かに依らず固定の部分
    /// # 引数
    /// - `get_input` 変換元のデータ
    /// - `twiddle`:回転因子
    fn fftin_core<F>(&self, get_input: F, twiddle: &[Complex<f64>]) -> Vec<Complex<f64>>
    where
        F: Fn(usize) -> Complex<f64>,
    {
        // 1) ビット逆転順で初期化
        let mut rec = Vec::with_capacity(self.size);
        for &idx in &self.rev_indices {
            rec.push(get_input(idx));
        }

        // 2) 蝶形演算ループ
        let mut span = self.size;
        let mut step = 1;
        while step < self.size {
            span /= 2;
            for s in (0..self.size).step_by(step * 2) {
                for i in 0..step {
                    let l = rec[s + i];
                    let r = rec[s + i + step] * twiddle[span * i];
                    rec[s + i] = l + r;
                    rec[s + i + step] = l - r;
                }
            }
            step *= 2;
        }

        rec
    }

    /// FFT と iFFT の共通処理部分。実数入力を Complex に変換して FFT を行う。
    /// # 引数
    /// - `&self`
    /// - `c`:変換元の実数データ
    /// - `twiddle`:回転因子
    fn fftin(&self, c: &[Complex<f64>], twiddle: &[Complex<f64>]) -> Vec<Complex<f64>> {
        self.fftin_core(|i| c[i], twiddle)
    }

    /// FFT と iFFT の共通処理部分。
    /// # 引数
    /// - `&self`
    /// - `c`:変換元の複素数データ
    /// - `twiddle`:回転因子
    fn fftin_real(&self, c: &[f64], twiddle: &[Complex<f64>]) -> Vec<Complex<f64>> {
        self.fftin_core(|i| Complex::new(c[i], 0.0), twiddle)
    }

    fn ifft(&self, f: &[Complex<f64>]) -> Vec<Complex<f64>> {
        self.fftin(f, &self.itwiddle)
            .into_iter()
            .map(|c| Complex::new(c.re / (self.size as f64), c.im / (self.size as f64)))
            .collect()
    }

    /// 実数データ f を入力として FFT を行い、周波数スペクトル (Complex 配列) を返す。
    /// # 引数
    /// - `&self`
    /// - `f`:変換元のデータ
    ///
    /// # 戻り値
    /// 周波数スペクトル (Complex 配列)
    pub fn fft_real(&self, f: &[f64]) -> Vec<Complex<f64>> {
        self.fftin_real(f, &self.twiddle)
    }
}

#[wasm_bindgen]
pub fn calc_spectrogram(size: usize, data: &[f64], window: &[f64]) -> Vec<f64> {
    let data_len = data.len();
    let window_size = window.len();
    assert!(window_size > 0,"window_size must be > 0");
    assert!(
        data_len >= size,
        "data length ({}) must be at least fft size ({})",
        data_len,
        size
    );
    let frame_count = (data_len - size) / window_size + 1;
    let freq_bins   = size / 2 + 1;
    let mut log_spec = Vec::with_capacity(frame_count * freq_bins);
    let f = FFT::new(size);
    for i in (0..data_len - size).step_by(window_size) {
        let windowed_data: Vec<f64> = data[i..i + size]
            .iter()
            .enumerate()
            .map(|(j, &t)| t * window[j % window_size])
            .collect();
        let spec: Vec<Complex<f64>> = f.fft_real(&windowed_data);
        //10log((re^2+im^2)^0.5)=5log(re^2+im^2)を求める
        for c in &spec {
            let sq = c.re * c.re + c.im * c.im;
            let db = 5.0 * sq.log10();
            log_spec.push(db);
        }
    }
    log_spec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn works_rev_bit() {
        let result = rev_bit(3, 1);
        assert_eq!(result, 4);
    }
    #[test]
    #[should_panic(expected = "size must be power of 2")]
    fn new_panics_on_non_power_of_two() {
        FFT::new(3);
    }
    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn new_panics_window_size_0() {
        let mut data: Vec<f64> = Vec::new();
        let mut window: Vec<f64> = Vec::new();
        data.push(1.0);
        data.push(1.0);
        data.push(1.0);
        data.push(1.0);
        calc_spectrogram(4, &data, &window);
    }
    #[test]
    #[should_panic(expected = "data length (2) must be at least fft size (4)")]
    fn new_panics_data_length_shorter() {
        let mut data: Vec<f64> = Vec::new();
        let mut window: Vec<f64> = Vec::new();
        data.push(1.0);
        data.push(1.0);
        window.push(1.0);
        window.push(1.0);
        window.push(1.0);
        window.push(1.0);
        calc_spectrogram(4, &data, &window);
    }
}

#[cfg(test)]
mod fft_tests {
    use super::*;
    const EPS: f64 = 1e-8;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn fft_real_constant_input() {
        let size = 8;
        let f = FFT::new(size);
        let data = vec![1.0_f64; size];
        let spec = f.fft_real(&data);

        assert_eq!(spec.len(), size);
        // DC 成分のみ N が返ってくる
        assert!(approx_eq(spec[0].re, size as f64));
        assert!(approx_eq(spec[0].im, 0.0));
        for k in 1..size {
            assert!(approx_eq(spec[k].re, 0.0));
            assert!(approx_eq(spec[k].im, 0.0));
        }
    }

    #[test]
    fn fft_real_impulse_input() {
        let size = 8;
        let f = FFT::new(size);
        let mut data = vec![0.0_f64; size];
        data[0] = 1.0;
        let spec = f.fft_real(&data);

        assert_eq!(spec.len(), size);
        // インパルスの DFT はすべての周波数で 1
        for k in 0..size {
            assert!(approx_eq(spec[k].re, 1.0));
            assert!(approx_eq(spec[k].im, 0.0));
        }
    }

    #[test]
    fn fft_then_ifft_returns_original() {
        // 長さは 2 のべき乗
        let size = 8;
        let fft = FFT::new(size);

        // 任意のテスト信号（ここでは 0,1,2,...,7）
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

        // FFT→iFFT の順に実行
        let spectrum = fft.fft_real(&data);
        let recovered = fft.ifft(&spectrum);

        // 復元値が元信号とほぼ一致するかチェック
        assert_eq!(recovered.len(), size);
        for (i, c) in recovered.iter().enumerate() {
            assert!(
                approx_eq(c.re, data[i]),
                "sample {}: got re={}, expected {}",
                i,
                c.re,
                data[i]
            );
            assert!(
                approx_eq(c.im, 0.0),
                "sample {}: imaginary part is not zero: {}",
                i,
                c.im
            );
        }
    }
}
