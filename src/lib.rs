use wasm_bindgen::prelude::*;
use num_complex::Complex;
use std::{ f32::consts::PI };

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
    twiddle: Vec<Complex<f32>>,

    /// iFFT 用の回転因子（`e^{+2πi * n / size}` の配列）
    itwiddle: Vec<Complex<f32>>,

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
        let k = (size as f32).log2() as usize;
        let t = (-2.0 * PI) / (size as f32);
        let it = (2.0 * PI) / (size as f32);
        let mut twiddle = Vec::with_capacity(size);
        let mut itwiddle = Vec::with_capacity(size);
        for i in 0..size {
            twiddle.push(Complex::from_polar(1.0, t * (i as f32)));
            itwiddle.push(Complex::from_polar(1.0, it * (i as f32)));
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
    fn fftin_core<F>(&self, get_input: F, twiddle: &[Complex<f32>]) -> Vec<Complex<f32>>
    where
        F: Fn(usize) -> Complex<f32>,
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
    fn fftin(&self, c: &[Complex<f32>], twiddle: &[Complex<f32>]) -> Vec<Complex<f32>> {
        self.fftin_core(|i| c[i], twiddle)
    }

    /// FFT と iFFT の共通処理部分。
    /// # 引数
    /// - `&self`
    /// - `c`:変換元の複素数データ
    /// - `twiddle`:回転因子
    fn fftin_real(&self, c: &[f32], twiddle: &[Complex<f32>]) -> Vec<Complex<f32>> {
        self.fftin_core(|i| Complex::new(c[i], 0.0), twiddle)
    }

    fn ifft(&self, f: &[Complex<f32>]) -> Vec<Complex<f32>> {
        self.fftin(f, &self.itwiddle)
            .into_iter()
            .map(|c| Complex::new(c.re / (self.size as f32), c.im / (self.size as f32)))
            .collect()
    }

    /// 実数データ f を入力として FFT を行い、周波数スペクトル (Complex 配列) を返す。
    /// # 引数
    /// - `&self`
    /// - `f`:変換元のデータ
    ///
    /// # 戻り値
    /// 周波数スペクトル (Complex 配列)
    pub fn fft_real(&self, f: &[f32]) -> Vec<Complex<f32>> {
        self.fftin_real(f, &self.twiddle)
    }

}

/// 実数に限定する代わりに高速に動作するFFT ライブラリを表す構造体
///
/// この構造体は、事前に計算した回転因子やビット逆転インデックスを保持する。
pub struct RealFft {
    full: FFT,                     // N-point FFT（今後の ifft でも使える）
    half: FFT,                     // N/2-point FFT
    rfft_twiddle: Vec<Complex<f32>>, // combine 用の e^{-2πi k / N}
}

impl RealFft {
    /// コンストラクタ
    ///
    /// # 引数
    /// - `size`:fftのサイズ。
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two());
        let full = FFT::new(size);
        let half = FFT::new(size / 2);

        let mut rfft_twiddle = Vec::with_capacity(size/2);
        for k in 0..(size/2) {
            let θ = -2.0 * PI * (k as f32) / (size as f32);
            rfft_twiddle.push(Complex::new(θ.cos(), θ.sin()));
        }

        RealFft { full, half, rfft_twiddle }
    }

    /// 実数データ f を入力として FFT を行い、周波数スペクトル (Complex 配列) を返す。
    /// # 引数
    /// - `&self`
    /// - `real`:実数限定
    ///
    /// # 戻り値
    /// 周波数スペクトル (Complex 配列)
    pub fn rfft(&self, real: &[f32]) -> Vec<Complex<f32>> {
        let n = self.full.size;
        let nh = n / 2;
        assert_eq!(real.len(), n);

        // 1) 実部＝偶数, 虚部＝奇数 でパック
        let mut buf = Vec::with_capacity(nh);
        for i in 0..nh {
            buf.push(Complex::new(real[2*i], real[2*i+1]));
        }

        // 2) 半長 FFT はキャッシュ済み
        let spec_half = self.half.fftin_core(|i| buf[i], &self.half.twiddle);

        // 3) combine
        let mut out = Vec::with_capacity(nh+1);
        let e0 = spec_half[0];
        out.push(Complex::new(e0.re + e0.im, 0.0));

        for k in 1..nh {
            let ek   = spec_half[k];
            let enk  = spec_half[nh - k].conj();
            let w    = self.rfft_twiddle[k];

            let sum   = ek + enk;
            let diff  = ek - enk;
            let cross = Complex::new(0.0, -1.0) * diff * w;

            out.push(Complex::new(
                (sum.re + cross.re) * 0.5,
                (sum.im + cross.im) * 0.5,
            ));
        }

        out.push(Complex::new(e0.re - e0.im, 0.0));
        out
    }
}



#[wasm_bindgen]
pub fn calc_spectrogram(size: usize, data: &[f32], window: &[f32]) -> Vec<f32> {
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
    let mut windowed_data = vec![0.0_f32; size];
    let f = FFT::new(size);
    for i in (0..data_len - size).step_by(window_size) {
        for j in 0..size {
            windowed_data[j] = data[i + j] * window[j % window_size];
        }
        let spec: Vec<Complex<f32>> = f.fft_real(&windowed_data);
        //10log((re^2+im^2)^0.5)=5log(re^2+im^2)を求める
        for c in &spec {
            let sq = c.re * c.re + c.im * c.im;
            let db = 5.0 * sq.log10();
            log_spec.push(db);
        }
    }
    log_spec
}

#[wasm_bindgen]
pub fn calc_spectrogram_with_rfft(size: usize, data: &[f32], window: &[f32]) -> Vec<f32> {
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
    let mut windowed_data = vec![0.0_f32; size];
    let f = RealFft::new(size);
    for i in (0..data_len - size).step_by(window_size) {
        for j in 0..size {
            windowed_data[j] = data[i + j] * window[j % window_size];
        }
        let spec: Vec<Complex<f32>> = f.rfft(&windowed_data);
        //10log((re^2+im^2)^0.5)=5log(re^2+im^2)を求める
        for c in &spec {
            let sq = c.re * c.re + c.im * c.im;
            let db = 5.0 * sq.log10();
            log_spec.push(db);
        }
    }
    log_spec
}


#[wasm_bindgen]
pub fn identity_array(size: usize, data: &[f32], window: &[f32]) -> Vec<f32> {
    // ただ入力をそのまま返すだけ
    data.to_vec()
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
        let mut data: Vec<f32> = Vec::new();
        let window: Vec<f32> = Vec::new();
        data.push(1.0);
        data.push(1.0);
        data.push(1.0);
        data.push(1.0);
        calc_spectrogram(4, &data, &window);
    }
    #[test]
    #[should_panic(expected = "data length (2) must be at least fft size (4)")]
    fn new_panics_data_length_shorter() {
        let mut data: Vec<f32> = Vec::new();
        let mut window: Vec<f32> = Vec::new();
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
    const EPS: f32 = 1e-8;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn fft_real_constant_input() {
        let size = 8;
        let f = FFT::new(size);
        let data = vec![1.0_f32; size];
        let spec = f.fft_real(&data);

        assert_eq!(spec.len(), size);
        // DC 成分のみ N が返ってくる
        assert!(approx_eq(spec[0].re, size as f32));
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
        let mut data = vec![0.0_f32; size];
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
    fn rfft_impulse_input() {
        let size = 8;
        let f = RealFft::new(size);
        let mut data = vec![0.0_f32; size];
        data[0] = 1.0;
        let spec = f.rfft(&data);
    
        // 出力長は N/2+1
        assert_eq!(spec.len(), size/2 + 1);
    
        // インパルスなら k=0…N/2 のすべてのビンが 1+0j
        for c in &spec {
            assert!(approx_eq(c.re, 1.0));
            assert!(approx_eq(c.im, 0.0));
        }
    }
    
}
