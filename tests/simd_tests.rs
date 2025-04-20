use fft_wasm_lib::FFT;
// 属性マクロを明示的に取り込む
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};
wasm_bindgen_test_configure!(run_in_browser);


#[wasm_bindgen_test]
fn fft_simd_matches_scalar() {
    let size = 8;
    let f = FFT::new(size);
    let data = vec![1.0_f32; size];
    let spec = f.fft_real(&data);
    // DC 成分だけ size, あとは 0 になるはず
    assert_eq!(spec.len(), size);
    assert!((spec[0].re - size as f32).abs() < 1e-8);
    for k in 1..size {
        assert!(spec[k].re.abs() < 1e-8);
        assert!(spec[k].im.abs() < 1e-8);
    }
}
