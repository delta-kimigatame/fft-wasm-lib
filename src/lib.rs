use wasm_bindgen::prelude::*;
use num_complex::Complex;
use std::{f64::consts::PI};

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

    /// FFT と iFFT の共通処理部分。実数入力を Complex に変換して FFT を行う。
    /// # 引数
    /// - `&self`
    /// - `c`:変換元データ
    /// - `twiddle`:回転因子
    fn fftin_real(&self, c: &[f64], twiddle: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let mut rec = vec![Complex::new(0.0, 0.0); self.size];
        for i in 0..self.size {
            rec[i] = Complex::new(c[self.rev_indices[i]], 0.0);
        }
        let mut t_factor = self.size;
        let mut Nh = 1;
        while Nh < self.size {
            t_factor /= 2;
            let mut s = 0;
            while s < self.size {
                for i in 0..Nh {
                    let l = rec[s + i];
                    let re = rec[s + i + Nh] * twiddle[t_factor * i];
                    rec[s + i] = l + re;
                    rec[s + i + Nh] = l - re;
                }
                s += Nh * 2;
            }
            Nh *= 2;
        }
        rec
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
    let mut log_spec:Vec<f64> = Vec::new();
    let f = FFT::new(size);
    for i in (0..data_len - size).step_by(window_size) {
        let windowed_data: Vec<f64> = data[i..i + size]
            .iter()
            .enumerate()
            .map(|(j, &t)| t * window[j % window_size])
            .collect();
        let spec:Vec<Complex<f64>>=f.fft_real(&windowed_data);
        //10log((re^2+im^2)^0.5)=5log(re^2+im^2)を求める
        for c in &spec{
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
}
