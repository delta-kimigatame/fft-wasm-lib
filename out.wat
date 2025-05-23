(module
 (type $0 (func (param i32 i32)))
 (type $1 (func (param i32 i32) (result i32)))
 (type $2 (func (param i32 i32 i32) (result i32)))
 (type $3 (func (param i32 i32 i32)))
 (type $4 (func (param i32)))
 (type $5 (func (param i32 i32 i32 i32) (result i32)))
 (type $6 (func (param f64) (result f64)))
 (type $7 (func (result i32 i32)))
 (type $8 (func))
 (type $9 (func (param i32) (result i32)))
 (type $10 (func (param i32 f64)))
 (type $11 (func (param i32 i32 i32 i32 i32)))
 (type $12 (func (param i32 i32 i32 i32)))
 (type $13 (func (param i32 i32 i32 i32 i32) (result i32 i32)))
 (type $14 (func (param i32 f64 f64)))
 (import "./fft_wasm_lib_bg.js" "__wbindgen_init_externref_table" (func $fimport$0))
 (global $global$0 (mut i32) (i32.const 1048576))
 (memory $0 17)
 (data $0 (i32.const 1048576) "/rustc/05f9846f893b09a1be1fc8560e33fc3c815cfecb/library/core/src/iter/adapters/step_by.rssrc/lib.rs\00Y\00\10\00\n\00\00\008\00\00\00\1b\00\00\00Y\00\10\00\n\00\00\009\00\00\00\1c\00\00\00Y\00\10\00\n\00\00\00>\00\00\00\1f\00\00\00Y\00\10\00\n\00\00\00;\00\00\00\15\00\00\00Y\00\10\00\n\00\00\00<\00\00\00\16\00\00\00size must be power of 2\00\b4\00\10\00\17\00\00\00Y\00\10\00\n\00\00\004\00\00\00\t\00\00\00Y\00\10\00\n\00\00\00w\00\00\00\17\00\00\00assertion failed: step != 0\00\00\00\10\00Y\00\00\00#\00\00\00\t\00\00\00Y\00\10\00\n\00\00\00\ae\00\00\00 \00\00\00Y\00\10\00\n\00\00\00\af\00\00\00 \00\00\00Y\00\10\00\n\00\00\00\af\00\00\001\00\00\00Y\00\10\00\n\00\00\00\85\00\00\00!\00\00\00Y\00\10\00\n\00\00\00\86\00\00\00!\00\00\00Y\00\10\00\n\00\00\00\88\00\00\00!\00\00\00Y\00\10\00\n\00\00\00\88\00\00\002\00\00\00Y\00\10\00\n\00\00\00\89\00\00\00!\00\00\00Y\00\10\00\n\00\00\00\89\00\00\006\00\00\00Y\00\10\00\n\00\00\00y\00\00\00\11\00\00\00Y\00\10\00\n\00\00\00\c9\00\00\00*\00\00\00window_size must be > 0\00\d0\01\10\00\17\00\00\00Y\00\10\00\n\00\00\00\e3\00\00\00\05\00\00\00data length () must be at least fft size ()\00\00\02\10\00\r\00\00\00\r\02\10\00\1d\00\00\00*\02\10\00\01\00\00\00Y\00\10\00\n\00\00\00\e4\00\00\00\05\00\00\00Y\00\10\00\n\00\00\00\ec\00\00\00\18\00\00\00Y\00\10\00\n\00\00\00\ef\00\00\00+\00\00\00/rustc/05f9846f893b09a1be1fc8560e33fc3c815cfecb/library/core/src/iter/traits/iterator.rst\02\10\00X\00\00\00\b3\07\00\00\t\00\00\00Y\00\10\00\n\00\00\00\f9\00\00\00\16\00\00\00/home/delta/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/wasm-bindgen-0.2.100/src/convert/slices.rs\00\ec\02\10\00k\00\00\00$\01\00\00\0e\00\00\00Lazy instance has previously been poisoned\00\00h\03\10\00*\00\00\00/home/delta/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/once_cell-1.21.3/src/lib.rs\9c\03\10\00\\\00\00\00\08\03\00\00\19\00\00\00reentrant init\00\00\08\04\10\00\0e\00\00\00\9c\03\10\00\\\00\00\00z\02\00\00\r\00\00\00/rustc/05f9846f893b09a1be1fc8560e33fc3c815cfecb/library/alloc/src/string.rs\000\04\10\00K\00\00\00\8d\05\00\00\1b\00\00\00/rustc/05f9846f893b09a1be1fc8560e33fc3c815cfecb/library/alloc/src/raw_vec.rs\8c\04\10\00L\00\00\00*\02\00\00\11\00\00\00\04\00\00\00\0c\00\00\00\04\00\00\00\05\00\00\00\06\00\00\00\07\00\00\00/rust/deps/dlmalloc-0.2.7/src/dlmalloc.rsassertion failed: psize >= size + min_overhead\00\00\05\10\00)\00\00\00\a8\04\00\00\t\00\00\00assertion failed: psize <= size + max_overhead\00\00\00\05\10\00)\00\00\00\ae\04\00\00\r\00\00\00memory allocation of  bytes failed\00\00\a8\05\10\00\15\00\00\00\bd\05\10\00\r\00\00\00library/std/src/alloc.rs\dc\05\10\00\18\00\00\00c\01\00\00\t\00\00\00\04\00\00\00\0c\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\04\00\00\00\t\00\00\00\00\00\00\00\08\00\00\00\04\00\00\00\n\00\00\00\0b\00\00\00\0c\00\00\00\r\00\00\00\0e\00\00\00\10\00\00\00\04\00\00\00\0f\00\00\00\10\00\00\00\11\00\00\00\12\00\00\00capacity overflow\00\00\00\\\06\10\00\11\00\00\00index out of bounds: the len is  but the index is \00\00x\06\10\00 \00\00\00\98\06\10\00\12\00\00\0000010203040506070809101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899 out of range for slice of length range end index \00\00\a6\07\10\00\10\00\00\00\84\07\10\00\"\00\00\00slice index starts at  but ends at \00\c8\07\10\00\16\00\00\00\de\07\10\00\r\00\00\00\03\00\00\00\04\00\00\00\04\00\00\00\06\00\00\00\83\f9\a2\00DNn\00\fc)\15\00\d1W\'\00\dd4\f5\00b\db\c0\00<\99\95\00A\90C\00cQ\fe\00\bb\de\ab\00\b7a\c5\00:n$\00\d2MB\00I\06\e0\00\t\ea.\00\1c\92\d1\00\eb\1d\fe\00)\b1\1c\00\e8>\a7\00\f55\82\00D\bb.\00\9c\e9\84\00\b4&p\00A~_\00\d6\919\00S\839\00\9c\f49\00\8b_\84\00(\f9\bd\00\f8\1f;\00\de\ff\97\00\0f\98\05\00\11/\ef\00\nZ\8b\00m\1fm\00\cf~6\00\t\cb\'\00FO\b7\00\9ef?\00-\ea_\00\ba\'u\00\e5\eb\c7\00={\f1\00\f79\07\00\92R\8a\00\fbk\ea\00\1f\b1_\00\08]\8d\000\03V\00{\fcF\00\f0\abk\00 \bc\cf\006\f4\9a\00\e3\a9\1d\00^a\91\00\08\1b\e6\00\85\99e\00\a0\14_\00\8d@h\00\80\d8\ff\00\'sM\00\06\061\00\caV\15\00\c9\a8s\00{\e2`\00k\8c\c0\00\00\00\00\00\00\00\00@\fb!\f9?\00\00\00\00-Dt>\00\00\00\80\98F\f8<\00\00\00`Q\ccx;\00\00\00\80\83\1b\f09\00\00\00@ %z8\00\00\00\80\"\82\e36\00\00\00\00\1d\f3i5")
 (data $1 (i32.const 1050992) "\02")
 (table $0 19 19 funcref)
 (table $1 128 externref)
 (elem $0 (table $0) (i32.const 1) func $43 $30 $19 $28 $20 $9 $44 $38 $39 $41 $21 $40 $45 $27 $17 $11 $14 $47)
 (export "memory" (memory $0))
 (export "calc_spectrogram" (func $25))
 (export "__wbindgen_export_0" (table $1))
 (export "__wbindgen_malloc" (func $24))
 (export "__wbindgen_free" (func $32))
 (export "__wbindgen_start" (func $fimport$0))
 (func $0 (param $0 i32) (result i32)
  (local $1 i32)
  (local $2 i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i32)
  (local $6 i32)
  (local $7 i32)
  (local $8 i32)
  (local $9 i32)
  (local $10 i64)
  (local $scratch i32)
  (global.set $global$0
   (local.tee $8
    (i32.sub
     (global.get $global$0)
     (i32.const 16)
    )
   )
  )
  (local.set $scratch
   (block $block (result i32)
    (block $block19
     (block $block20
      (block $block1
       (block $block16
        (block $block4
         (block $block2
          (if
           (i32.ge_u
            (local.get $0)
            (i32.const 245)
           )
           (then
            (drop
             (br_if $block
              (i32.const 0)
              (i32.gt_u
               (local.get $0)
               (i32.const -65588)
              )
             )
            )
            (local.set $5
             (i32.and
              (local.tee $1
               (i32.add
                (local.get $0)
                (i32.const 11)
               )
              )
              (i32.const -8)
             )
            )
            (br_if $block1
             (i32.eqz
              (local.tee $9
               (i32.load
                (i32.const 1051440)
               )
              )
             )
            )
            (local.set $7
             (i32.const 31)
            )
            (local.set $4
             (i32.sub
              (i32.const 0)
              (local.get $5)
             )
            )
            (if
             (i32.le_u
              (local.get $0)
              (i32.const 16777204)
             )
             (then
              (local.set $7
               (i32.add
                (i32.sub
                 (i32.and
                  (i32.shr_u
                   (local.get $5)
                   (i32.sub
                    (i32.const 6)
                    (local.tee $0
                     (i32.clz
                      (i32.shr_u
                       (local.get $1)
                       (i32.const 8)
                      )
                     )
                    )
                   )
                  )
                  (i32.const 1)
                 )
                 (i32.shl
                  (local.get $0)
                  (i32.const 1)
                 )
                )
                (i32.const 62)
               )
              )
             )
            )
            (if
             (i32.eqz
              (local.tee $1
               (i32.load
                (i32.add
                 (i32.shl
                  (local.get $7)
                  (i32.const 2)
                 )
                 (i32.const 1051028)
                )
               )
              )
             )
             (then
              (local.set $0
               (i32.const 0)
              )
              (br $block2)
             )
            )
            (local.set $0
             (i32.const 0)
            )
            (local.set $3
             (i32.shl
              (local.get $5)
              (select
               (i32.sub
                (i32.const 25)
                (i32.shr_u
                 (local.get $7)
                 (i32.const 1)
                )
               )
               (i32.const 0)
               (i32.ne
                (local.get $7)
                (i32.const 31)
               )
              )
             )
            )
            (loop $label
             (block $block3
              (br_if $block3
               (i32.lt_u
                (local.tee $6
                 (i32.and
                  (i32.load offset=4
                   (local.get $1)
                  )
                  (i32.const -8)
                 )
                )
                (local.get $5)
               )
              )
              (br_if $block3
               (i32.ge_u
                (local.tee $6
                 (i32.sub
                  (local.get $6)
                  (local.get $5)
                 )
                )
                (local.get $4)
               )
              )
              (local.set $2
               (local.get $1)
              )
              (br_if $block3
               (local.tee $4
                (local.get $6)
               )
              )
              (local.set $4
               (i32.const 0)
              )
              (local.set $0
               (local.get $1)
              )
              (br $block4)
             )
             (local.set $0
              (select
               (select
                (local.tee $6
                 (i32.load offset=20
                  (local.get $1)
                 )
                )
                (local.get $0)
                (i32.ne
                 (local.get $6)
                 (local.tee $1
                  (i32.load
                   (i32.add
                    (i32.add
                     (local.get $1)
                     (i32.and
                      (i32.shr_u
                       (local.get $3)
                       (i32.const 29)
                      )
                      (i32.const 4)
                     )
                    )
                    (i32.const 16)
                   )
                  )
                 )
                )
               )
               (local.get $0)
               (local.get $6)
              )
             )
             (local.set $3
              (i32.shl
               (local.get $3)
               (i32.const 1)
              )
             )
             (br_if $label
              (local.get $1)
             )
            )
            (br $block2)
           )
          )
          (if
           (i32.and
            (local.tee $1
             (i32.shr_u
              (local.tee $2
               (i32.load
                (i32.const 1051436)
               )
              )
              (local.tee $0
               (i32.shr_u
                (local.tee $5
                 (select
                  (i32.const 16)
                  (i32.and
                   (i32.add
                    (local.get $0)
                    (i32.const 11)
                   )
                   (i32.const 504)
                  )
                  (i32.lt_u
                   (local.get $0)
                   (i32.const 11)
                  )
                 )
                )
                (i32.const 3)
               )
              )
             )
            )
            (i32.const 3)
           )
           (then
            (block $block5
             (if
              (i32.ne
               (local.tee $3
                (i32.add
                 (local.tee $0
                  (i32.shl
                   (local.tee $6
                    (i32.add
                     (i32.and
                      (i32.xor
                       (local.get $1)
                       (i32.const -1)
                      )
                      (i32.const 1)
                     )
                     (local.get $0)
                    )
                   )
                   (i32.const 3)
                  )
                 )
                 (i32.const 1051172)
                )
               )
               (local.tee $4
                (i32.load offset=8
                 (local.tee $1
                  (i32.load
                   (i32.add
                    (local.get $0)
                    (i32.const 1051180)
                   )
                  )
                 )
                )
               )
              )
              (then
               (i32.store offset=12
                (local.get $4)
                (local.get $3)
               )
               (i32.store offset=8
                (local.get $3)
                (local.get $4)
               )
               (br $block5)
              )
             )
             (i32.store
              (i32.const 1051436)
              (i32.and
               (local.get $2)
               (i32.rotl
                (i32.const -2)
                (local.get $6)
               )
              )
             )
            )
            (i32.store offset=4
             (local.get $1)
             (i32.or
              (local.get $0)
              (i32.const 3)
             )
            )
            (i32.store offset=4
             (local.tee $0
              (i32.add
               (local.get $0)
               (local.get $1)
              )
             )
             (i32.or
              (i32.load offset=4
               (local.get $0)
              )
              (i32.const 1)
             )
            )
            (br $block
             (i32.add
              (local.get $1)
              (i32.const 8)
             )
            )
           )
          )
          (br_if $block1
           (i32.le_u
            (local.get $5)
            (i32.load
             (i32.const 1051444)
            )
           )
          )
          (block $block9
           (block $block10
            (if
             (i32.eqz
              (local.get $1)
             )
             (then
              (br_if $block1
               (i32.eqz
                (local.tee $0
                 (i32.load
                  (i32.const 1051440)
                 )
                )
               )
              )
              (local.set $4
               (i32.sub
                (i32.and
                 (i32.load offset=4
                  (local.tee $2
                   (i32.load
                    (i32.add
                     (i32.shl
                      (i32.ctz
                       (local.get $0)
                      )
                      (i32.const 2)
                     )
                     (i32.const 1051028)
                    )
                   )
                  )
                 )
                 (i32.const -8)
                )
                (local.get $5)
               )
              )
              (local.set $1
               (local.get $2)
              )
              (loop $label2
               (block $block6
                (br_if $block6
                 (local.tee $0
                  (i32.load offset=16
                   (local.get $2)
                  )
                 )
                )
                (br_if $block6
                 (local.tee $0
                  (i32.load offset=20
                   (local.get $2)
                  )
                 )
                )
                (local.set $7
                 (i32.load offset=24
                  (local.get $1)
                 )
                )
                (block $block8
                 (block $block7
                  (if
                   (i32.eq
                    (local.get $1)
                    (local.tee $0
                     (i32.load offset=12
                      (local.get $1)
                     )
                    )
                   )
                   (then
                    (br_if $block7
                     (local.tee $2
                      (i32.load
                       (i32.add
                        (local.get $1)
                        (select
                         (i32.const 20)
                         (i32.const 16)
                         (local.tee $0
                          (i32.load offset=20
                           (local.get $1)
                          )
                         )
                        )
                       )
                      )
                     )
                    )
                    (local.set $0
                     (i32.const 0)
                    )
                    (br $block8)
                   )
                  )
                  (i32.store offset=12
                   (local.tee $2
                    (i32.load offset=8
                     (local.get $1)
                    )
                   )
                   (local.get $0)
                  )
                  (i32.store offset=8
                   (local.get $0)
                   (local.get $2)
                  )
                  (br $block8)
                 )
                 (local.set $3
                  (select
                   (i32.add
                    (local.get $1)
                    (i32.const 20)
                   )
                   (i32.add
                    (local.get $1)
                    (i32.const 16)
                   )
                   (local.get $0)
                  )
                 )
                 (loop $label1
                  (local.set $6
                   (local.get $3)
                  )
                  (local.set $3
                   (select
                    (i32.add
                     (local.tee $0
                      (local.get $2)
                     )
                     (i32.const 20)
                    )
                    (i32.add
                     (local.get $0)
                     (i32.const 16)
                    )
                    (local.tee $2
                     (i32.load offset=20
                      (local.get $0)
                     )
                    )
                   )
                  )
                  (br_if $label1
                   (local.tee $2
                    (i32.load
                     (i32.add
                      (local.get $0)
                      (select
                       (i32.const 20)
                       (i32.const 16)
                       (local.get $2)
                      )
                     )
                    )
                   )
                  )
                 )
                 (i32.store
                  (local.get $6)
                  (i32.const 0)
                 )
                )
                (br_if $block9
                 (i32.eqz
                  (local.get $7)
                 )
                )
                (if
                 (i32.ne
                  (local.get $1)
                  (i32.load
                   (local.tee $2
                    (i32.add
                     (i32.shl
                      (i32.load offset=28
                       (local.get $1)
                      )
                      (i32.const 2)
                     )
                     (i32.const 1051028)
                    )
                   )
                  )
                 )
                 (then
                  (i32.store
                   (i32.add
                    (local.get $7)
                    (select
                     (i32.const 16)
                     (i32.const 20)
                     (i32.eq
                      (i32.load offset=16
                       (local.get $7)
                      )
                      (local.get $1)
                     )
                    )
                   )
                   (local.get $0)
                  )
                  (br_if $block9
                   (i32.eqz
                    (local.get $0)
                   )
                  )
                  (br $block10)
                 )
                )
                (i32.store
                 (local.get $2)
                 (local.get $0)
                )
                (br_if $block10
                 (local.get $0)
                )
                (i32.store
                 (i32.const 1051440)
                 (i32.and
                  (i32.load
                   (i32.const 1051440)
                  )
                  (i32.rotl
                   (i32.const -2)
                   (i32.load offset=28
                    (local.get $1)
                   )
                  )
                 )
                )
                (br $block9)
               )
               (local.set $4
                (select
                 (local.tee $2
                  (i32.sub
                   (i32.and
                    (i32.load offset=4
                     (local.get $0)
                    )
                    (i32.const -8)
                   )
                   (local.get $5)
                  )
                 )
                 (local.get $4)
                 (local.tee $2
                  (i32.lt_u
                   (local.get $2)
                   (local.get $4)
                  )
                 )
                )
               )
               (local.set $1
                (select
                 (local.get $0)
                 (local.get $1)
                 (local.get $2)
                )
               )
               (local.set $2
                (local.get $0)
               )
               (br $label2)
              )
              (unreachable)
             )
            )
            (block $block11
             (if
              (i32.ne
               (local.tee $3
                (i32.add
                 (local.tee $1
                  (i32.shl
                   (local.tee $6
                    (i32.ctz
                     (i32.and
                      (i32.or
                       (local.tee $3
                        (i32.shl
                         (i32.const 2)
                         (local.get $0)
                        )
                       )
                       (i32.sub
                        (i32.const 0)
                        (local.get $3)
                       )
                      )
                      (i32.shl
                       (local.get $1)
                       (local.get $0)
                      )
                     )
                    )
                   )
                   (i32.const 3)
                  )
                 )
                 (i32.const 1051172)
                )
               )
               (local.tee $4
                (i32.load offset=8
                 (local.tee $0
                  (i32.load
                   (i32.add
                    (local.get $1)
                    (i32.const 1051180)
                   )
                  )
                 )
                )
               )
              )
              (then
               (i32.store offset=12
                (local.get $4)
                (local.get $3)
               )
               (i32.store offset=8
                (local.get $3)
                (local.get $4)
               )
               (br $block11)
              )
             )
             (i32.store
              (i32.const 1051436)
              (i32.and
               (local.get $2)
               (i32.rotl
                (i32.const -2)
                (local.get $6)
               )
              )
             )
            )
            (i32.store offset=4
             (local.get $0)
             (i32.or
              (local.get $5)
              (i32.const 3)
             )
            )
            (i32.store offset=4
             (local.tee $6
              (i32.add
               (local.get $0)
               (local.get $5)
              )
             )
             (i32.or
              (local.tee $3
               (i32.sub
                (local.get $1)
                (local.get $5)
               )
              )
              (i32.const 1)
             )
            )
            (i32.store
             (i32.add
              (local.get $0)
              (local.get $1)
             )
             (local.get $3)
            )
            (if
             (local.tee $4
              (i32.load
               (i32.const 1051444)
              )
             )
             (then
              (local.set $1
               (i32.add
                (i32.and
                 (local.get $4)
                 (i32.const -8)
                )
                (i32.const 1051172)
               )
              )
              (local.set $2
               (i32.load
                (i32.const 1051452)
               )
              )
              (local.set $4
               (block $block12 (result i32)
                (if
                 (i32.eqz
                  (i32.and
                   (local.tee $5
                    (i32.load
                     (i32.const 1051436)
                    )
                   )
                   (local.tee $4
                    (i32.shl
                     (i32.const 1)
                     (i32.shr_u
                      (local.get $4)
                      (i32.const 3)
                     )
                    )
                   )
                  )
                 )
                 (then
                  (i32.store
                   (i32.const 1051436)
                   (i32.or
                    (local.get $4)
                    (local.get $5)
                   )
                  )
                  (br $block12
                   (local.get $1)
                  )
                 )
                )
                (i32.load offset=8
                 (local.get $1)
                )
               )
              )
              (i32.store offset=8
               (local.get $1)
               (local.get $2)
              )
              (i32.store offset=12
               (local.get $4)
               (local.get $2)
              )
              (i32.store offset=12
               (local.get $2)
               (local.get $1)
              )
              (i32.store offset=8
               (local.get $2)
               (local.get $4)
              )
             )
            )
            (i32.store
             (i32.const 1051452)
             (local.get $6)
            )
            (i32.store
             (i32.const 1051444)
             (local.get $3)
            )
            (br $block
             (i32.add
              (local.get $0)
              (i32.const 8)
             )
            )
           )
           (i32.store offset=24
            (local.get $0)
            (local.get $7)
           )
           (if
            (local.tee $2
             (i32.load offset=16
              (local.get $1)
             )
            )
            (then
             (i32.store offset=16
              (local.get $0)
              (local.get $2)
             )
             (i32.store offset=24
              (local.get $2)
              (local.get $0)
             )
            )
           )
           (br_if $block9
            (i32.eqz
             (local.tee $2
              (i32.load offset=20
               (local.get $1)
              )
             )
            )
           )
           (i32.store offset=20
            (local.get $0)
            (local.get $2)
           )
           (i32.store offset=24
            (local.get $2)
            (local.get $0)
           )
          )
          (block $block15
           (block $block13
            (if
             (i32.ge_u
              (local.get $4)
              (i32.const 16)
             )
             (then
              (i32.store offset=4
               (local.get $1)
               (i32.or
                (local.get $5)
                (i32.const 3)
               )
              )
              (i32.store offset=4
               (local.tee $3
                (i32.add
                 (local.get $1)
                 (local.get $5)
                )
               )
               (i32.or
                (local.get $4)
                (i32.const 1)
               )
              )
              (i32.store
               (i32.add
                (local.get $3)
                (local.get $4)
               )
               (local.get $4)
              )
              (br_if $block13
               (i32.eqz
                (local.tee $6
                 (i32.load
                  (i32.const 1051444)
                 )
                )
               )
              )
              (local.set $0
               (i32.add
                (i32.and
                 (local.get $6)
                 (i32.const -8)
                )
                (i32.const 1051172)
               )
              )
              (local.set $2
               (i32.load
                (i32.const 1051452)
               )
              )
              (local.set $6
               (block $block14 (result i32)
                (if
                 (i32.eqz
                  (i32.and
                   (local.tee $5
                    (i32.load
                     (i32.const 1051436)
                    )
                   )
                   (local.tee $6
                    (i32.shl
                     (i32.const 1)
                     (i32.shr_u
                      (local.get $6)
                      (i32.const 3)
                     )
                    )
                   )
                  )
                 )
                 (then
                  (i32.store
                   (i32.const 1051436)
                   (i32.or
                    (local.get $5)
                    (local.get $6)
                   )
                  )
                  (br $block14
                   (local.get $0)
                  )
                 )
                )
                (i32.load offset=8
                 (local.get $0)
                )
               )
              )
              (i32.store offset=8
               (local.get $0)
               (local.get $2)
              )
              (i32.store offset=12
               (local.get $6)
               (local.get $2)
              )
              (i32.store offset=12
               (local.get $2)
               (local.get $0)
              )
              (i32.store offset=8
               (local.get $2)
               (local.get $6)
              )
              (br $block13)
             )
            )
            (i32.store offset=4
             (local.get $1)
             (i32.or
              (local.tee $0
               (i32.add
                (local.get $4)
                (local.get $5)
               )
              )
              (i32.const 3)
             )
            )
            (i32.store offset=4
             (local.tee $0
              (i32.add
               (local.get $0)
               (local.get $1)
              )
             )
             (i32.or
              (i32.load offset=4
               (local.get $0)
              )
              (i32.const 1)
             )
            )
            (br $block15)
           )
           (i32.store
            (i32.const 1051452)
            (local.get $3)
           )
           (i32.store
            (i32.const 1051444)
            (local.get $4)
           )
          )
          (br $block
           (i32.add
            (local.get $1)
            (i32.const 8)
           )
          )
         )
         (if
          (i32.eqz
           (i32.or
            (local.get $0)
            (local.get $2)
           )
          )
          (then
           (local.set $2
            (i32.const 0)
           )
           (br_if $block1
            (i32.eqz
             (local.tee $0
              (i32.and
               (i32.or
                (local.tee $0
                 (i32.shl
                  (i32.const 2)
                  (local.get $7)
                 )
                )
                (i32.sub
                 (i32.const 0)
                 (local.get $0)
                )
               )
               (local.get $9)
              )
             )
            )
           )
           (local.set $0
            (i32.load
             (i32.add
              (i32.shl
               (i32.ctz
                (local.get $0)
               )
               (i32.const 2)
              )
              (i32.const 1051028)
             )
            )
           )
          )
         )
         (br_if $block16
          (i32.eqz
           (local.get $0)
          )
         )
        )
        (loop $label3
         (local.set $9
          (select
           (local.get $0)
           (local.get $2)
           (local.tee $7
            (i32.lt_u
             (local.tee $6
              (i32.sub
               (local.tee $3
                (i32.and
                 (i32.load offset=4
                  (local.get $0)
                 )
                 (i32.const -8)
                )
               )
               (local.get $5)
              )
             )
             (local.get $4)
            )
           )
          )
         )
         (if
          (i32.eqz
           (local.tee $1
            (i32.load offset=16
             (local.get $0)
            )
           )
          )
          (then
           (local.set $1
            (i32.load offset=20
             (local.get $0)
            )
           )
          )
         )
         (local.set $2
          (select
           (local.get $2)
           (local.get $9)
           (local.tee $0
            (i32.lt_u
             (local.get $3)
             (local.get $5)
            )
           )
          )
         )
         (local.set $4
          (select
           (local.get $4)
           (select
            (local.get $6)
            (local.get $4)
            (local.get $7)
           )
           (local.get $0)
          )
         )
         (br_if $label3
          (local.tee $0
           (local.get $1)
          )
         )
        )
       )
       (br_if $block1
        (i32.eqz
         (local.get $2)
        )
       )
       (br_if $block1
        (i32.and
         (i32.le_u
          (local.get $5)
          (local.tee $0
           (i32.load
            (i32.const 1051444)
           )
          )
         )
         (i32.ge_u
          (local.get $4)
          (i32.sub
           (local.get $0)
           (local.get $5)
          )
         )
        )
       )
       (local.set $7
        (i32.load offset=24
         (local.get $2)
        )
       )
       (block $block18
        (block $block17
         (if
          (i32.eq
           (local.get $2)
           (local.tee $0
            (i32.load offset=12
             (local.get $2)
            )
           )
          )
          (then
           (br_if $block17
            (local.tee $1
             (i32.load
              (i32.add
               (local.get $2)
               (select
                (i32.const 20)
                (i32.const 16)
                (local.tee $0
                 (i32.load offset=20
                  (local.get $2)
                 )
                )
               )
              )
             )
            )
           )
           (local.set $0
            (i32.const 0)
           )
           (br $block18)
          )
         )
         (i32.store offset=12
          (local.tee $1
           (i32.load offset=8
            (local.get $2)
           )
          )
          (local.get $0)
         )
         (i32.store offset=8
          (local.get $0)
          (local.get $1)
         )
         (br $block18)
        )
        (local.set $3
         (select
          (i32.add
           (local.get $2)
           (i32.const 20)
          )
          (i32.add
           (local.get $2)
           (i32.const 16)
          )
          (local.get $0)
         )
        )
        (loop $label4
         (local.set $6
          (local.get $3)
         )
         (local.set $3
          (select
           (i32.add
            (local.tee $0
             (local.get $1)
            )
            (i32.const 20)
           )
           (i32.add
            (local.get $0)
            (i32.const 16)
           )
           (local.tee $1
            (i32.load offset=20
             (local.get $0)
            )
           )
          )
         )
         (br_if $label4
          (local.tee $1
           (i32.load
            (i32.add
             (local.get $0)
             (select
              (i32.const 20)
              (i32.const 16)
              (local.get $1)
             )
            )
           )
          )
         )
        )
        (i32.store
         (local.get $6)
         (i32.const 0)
        )
       )
       (br_if $block19
        (i32.eqz
         (local.get $7)
        )
       )
       (if
        (i32.ne
         (local.get $2)
         (i32.load
          (local.tee $1
           (i32.add
            (i32.shl
             (i32.load offset=28
              (local.get $2)
             )
             (i32.const 2)
            )
            (i32.const 1051028)
           )
          )
         )
        )
        (then
         (i32.store
          (i32.add
           (local.get $7)
           (select
            (i32.const 16)
            (i32.const 20)
            (i32.eq
             (i32.load offset=16
              (local.get $7)
             )
             (local.get $2)
            )
           )
          )
          (local.get $0)
         )
         (br_if $block19
          (i32.eqz
           (local.get $0)
          )
         )
         (br $block20)
        )
       )
       (i32.store
        (local.get $1)
        (local.get $0)
       )
       (br_if $block20
        (local.get $0)
       )
       (i32.store
        (i32.const 1051440)
        (i32.and
         (i32.load
          (i32.const 1051440)
         )
         (i32.rotl
          (i32.const -2)
          (i32.load offset=28
           (local.get $2)
          )
         )
        )
       )
       (br $block19)
      )
      (block $block23
       (block $block31
        (block $block30
         (block $block29
          (block $block24
           (if
            (i32.gt_u
             (local.get $5)
             (local.tee $1
              (i32.load
               (i32.const 1051444)
              )
             )
            )
            (then
             (if
              (i32.ge_u
               (local.get $5)
               (local.tee $0
                (i32.load
                 (i32.const 1051448)
                )
               )
              )
              (then
               (local.set $0
                (memory.grow
                 (i32.shr_u
                  (local.tee $2
                   (i32.and
                    (i32.add
                     (local.get $5)
                     (i32.const 65583)
                    )
                    (i32.const -65536)
                   )
                  )
                  (i32.const 16)
                 )
                )
               )
               (i32.store offset=8
                (local.tee $1
                 (i32.add
                  (local.get $8)
                  (i32.const 4)
                 )
                )
                (i32.const 0)
               )
               (i32.store offset=4
                (local.get $1)
                (select
                 (i32.const 0)
                 (i32.and
                  (local.get $2)
                  (i32.const -65536)
                 )
                 (local.tee $2
                  (i32.eq
                   (local.get $0)
                   (i32.const -1)
                  )
                 )
                )
               )
               (i32.store
                (local.get $1)
                (select
                 (i32.const 0)
                 (i32.shl
                  (local.get $0)
                  (i32.const 16)
                 )
                 (local.get $2)
                )
               )
               (drop
                (br_if $block
                 (i32.const 0)
                 (i32.eqz
                  (local.tee $1
                   (i32.load offset=4
                    (local.get $8)
                   )
                  )
                 )
                )
               )
               (local.set $6
                (i32.load offset=12
                 (local.get $8)
                )
               )
               (i32.store
                (i32.const 1051460)
                (local.tee $0
                 (i32.add
                  (local.tee $4
                   (i32.load offset=8
                    (local.get $8)
                   )
                  )
                  (i32.load
                   (i32.const 1051460)
                  )
                 )
                )
               )
               (i32.store
                (i32.const 1051464)
                (select
                 (local.get $0)
                 (local.tee $2
                  (i32.load
                   (i32.const 1051464)
                  )
                 )
                 (i32.gt_u
                  (local.get $0)
                  (local.get $2)
                 )
                )
               )
               (block $block22
                (block $block21
                 (if
                  (local.tee $2
                   (i32.load
                    (i32.const 1051456)
                   )
                  )
                  (then
                   (local.set $0
                    (i32.const 1051156)
                   )
                   (loop $label5
                    (br_if $block21
                     (i32.eq
                      (local.get $1)
                      (i32.add
                       (local.tee $3
                        (i32.load
                         (local.get $0)
                        )
                       )
                       (local.tee $7
                        (i32.load offset=4
                         (local.get $0)
                        )
                       )
                      )
                     )
                    )
                    (br_if $label5
                     (local.tee $0
                      (i32.load offset=8
                       (local.get $0)
                      )
                     )
                    )
                   )
                   (br $block22)
                  )
                 )
                 (if
                  (i32.eqz
                   (select
                    (local.tee $0
                     (i32.load
                      (i32.const 1051472)
                     )
                    )
                    (i32.const 0)
                    (i32.le_u
                     (local.get $0)
                     (local.get $1)
                    )
                   )
                  )
                  (then
                   (i32.store
                    (i32.const 1051472)
                    (local.get $1)
                   )
                  )
                 )
                 (i32.store
                  (i32.const 1051476)
                  (i32.const 4095)
                 )
                 (i32.store
                  (i32.const 1051168)
                  (local.get $6)
                 )
                 (i32.store
                  (i32.const 1051160)
                  (local.get $4)
                 )
                 (i32.store
                  (i32.const 1051156)
                  (local.get $1)
                 )
                 (i32.store
                  (i32.const 1051184)
                  (i32.const 1051172)
                 )
                 (i32.store
                  (i32.const 1051192)
                  (i32.const 1051180)
                 )
                 (i32.store
                  (i32.const 1051180)
                  (i32.const 1051172)
                 )
                 (i32.store
                  (i32.const 1051200)
                  (i32.const 1051188)
                 )
                 (i32.store
                  (i32.const 1051188)
                  (i32.const 1051180)
                 )
                 (i32.store
                  (i32.const 1051208)
                  (i32.const 1051196)
                 )
                 (i32.store
                  (i32.const 1051196)
                  (i32.const 1051188)
                 )
                 (i32.store
                  (i32.const 1051216)
                  (i32.const 1051204)
                 )
                 (i32.store
                  (i32.const 1051204)
                  (i32.const 1051196)
                 )
                 (i32.store
                  (i32.const 1051224)
                  (i32.const 1051212)
                 )
                 (i32.store
                  (i32.const 1051212)
                  (i32.const 1051204)
                 )
                 (i32.store
                  (i32.const 1051232)
                  (i32.const 1051220)
                 )
                 (i32.store
                  (i32.const 1051220)
                  (i32.const 1051212)
                 )
                 (i32.store
                  (i32.const 1051240)
                  (i32.const 1051228)
                 )
                 (i32.store
                  (i32.const 1051228)
                  (i32.const 1051220)
                 )
                 (i32.store
                  (i32.const 1051248)
                  (i32.const 1051236)
                 )
                 (i32.store
                  (i32.const 1051236)
                  (i32.const 1051228)
                 )
                 (i32.store
                  (i32.const 1051244)
                  (i32.const 1051236)
                 )
                 (i32.store
                  (i32.const 1051256)
                  (i32.const 1051244)
                 )
                 (i32.store
                  (i32.const 1051252)
                  (i32.const 1051244)
                 )
                 (i32.store
                  (i32.const 1051264)
                  (i32.const 1051252)
                 )
                 (i32.store
                  (i32.const 1051260)
                  (i32.const 1051252)
                 )
                 (i32.store
                  (i32.const 1051272)
                  (i32.const 1051260)
                 )
                 (i32.store
                  (i32.const 1051268)
                  (i32.const 1051260)
                 )
                 (i32.store
                  (i32.const 1051280)
                  (i32.const 1051268)
                 )
                 (i32.store
                  (i32.const 1051276)
                  (i32.const 1051268)
                 )
                 (i32.store
                  (i32.const 1051288)
                  (i32.const 1051276)
                 )
                 (i32.store
                  (i32.const 1051284)
                  (i32.const 1051276)
                 )
                 (i32.store
                  (i32.const 1051296)
                  (i32.const 1051284)
                 )
                 (i32.store
                  (i32.const 1051292)
                  (i32.const 1051284)
                 )
                 (i32.store
                  (i32.const 1051304)
                  (i32.const 1051292)
                 )
                 (i32.store
                  (i32.const 1051300)
                  (i32.const 1051292)
                 )
                 (i32.store
                  (i32.const 1051312)
                  (i32.const 1051300)
                 )
                 (i32.store
                  (i32.const 1051320)
                  (i32.const 1051308)
                 )
                 (i32.store
                  (i32.const 1051308)
                  (i32.const 1051300)
                 )
                 (i32.store
                  (i32.const 1051328)
                  (i32.const 1051316)
                 )
                 (i32.store
                  (i32.const 1051316)
                  (i32.const 1051308)
                 )
                 (i32.store
                  (i32.const 1051336)
                  (i32.const 1051324)
                 )
                 (i32.store
                  (i32.const 1051324)
                  (i32.const 1051316)
                 )
                 (i32.store
                  (i32.const 1051344)
                  (i32.const 1051332)
                 )
                 (i32.store
                  (i32.const 1051332)
                  (i32.const 1051324)
                 )
                 (i32.store
                  (i32.const 1051352)
                  (i32.const 1051340)
                 )
                 (i32.store
                  (i32.const 1051340)
                  (i32.const 1051332)
                 )
                 (i32.store
                  (i32.const 1051360)
                  (i32.const 1051348)
                 )
                 (i32.store
                  (i32.const 1051348)
                  (i32.const 1051340)
                 )
                 (i32.store
                  (i32.const 1051368)
                  (i32.const 1051356)
                 )
                 (i32.store
                  (i32.const 1051356)
                  (i32.const 1051348)
                 )
                 (i32.store
                  (i32.const 1051376)
                  (i32.const 1051364)
                 )
                 (i32.store
                  (i32.const 1051364)
                  (i32.const 1051356)
                 )
                 (i32.store
                  (i32.const 1051384)
                  (i32.const 1051372)
                 )
                 (i32.store
                  (i32.const 1051372)
                  (i32.const 1051364)
                 )
                 (i32.store
                  (i32.const 1051392)
                  (i32.const 1051380)
                 )
                 (i32.store
                  (i32.const 1051380)
                  (i32.const 1051372)
                 )
                 (i32.store
                  (i32.const 1051400)
                  (i32.const 1051388)
                 )
                 (i32.store
                  (i32.const 1051388)
                  (i32.const 1051380)
                 )
                 (i32.store
                  (i32.const 1051408)
                  (i32.const 1051396)
                 )
                 (i32.store
                  (i32.const 1051396)
                  (i32.const 1051388)
                 )
                 (i32.store
                  (i32.const 1051416)
                  (i32.const 1051404)
                 )
                 (i32.store
                  (i32.const 1051404)
                  (i32.const 1051396)
                 )
                 (i32.store
                  (i32.const 1051424)
                  (i32.const 1051412)
                 )
                 (i32.store
                  (i32.const 1051412)
                  (i32.const 1051404)
                 )
                 (i32.store
                  (i32.const 1051432)
                  (i32.const 1051420)
                 )
                 (i32.store
                  (i32.const 1051420)
                  (i32.const 1051412)
                 )
                 (i32.store
                  (i32.const 1051456)
                  (local.tee $2
                   (i32.sub
                    (local.tee $0
                     (i32.and
                      (i32.add
                       (local.get $1)
                       (i32.const 15)
                      )
                      (i32.const -8)
                     )
                    )
                    (i32.const 8)
                   )
                  )
                 )
                 (i32.store
                  (i32.const 1051428)
                  (i32.const 1051420)
                 )
                 (i32.store
                  (i32.const 1051448)
                  (local.tee $0
                   (i32.add
                    (i32.add
                     (local.tee $3
                      (i32.sub
                       (local.get $4)
                       (i32.const 40)
                      )
                     )
                     (i32.sub
                      (local.get $1)
                      (local.get $0)
                     )
                    )
                    (i32.const 8)
                   )
                  )
                 )
                 (i32.store offset=4
                  (local.get $2)
                  (i32.or
                   (local.get $0)
                   (i32.const 1)
                  )
                 )
                 (i32.store offset=4
                  (i32.add
                   (local.get $1)
                   (local.get $3)
                  )
                  (i32.const 40)
                 )
                 (i32.store
                  (i32.const 1051468)
                  (i32.const 2097152)
                 )
                 (br $block23)
                )
                (br_if $block22
                 (i32.or
                  (i32.lt_u
                   (local.get $2)
                   (local.get $3)
                  )
                  (i32.le_u
                   (local.get $1)
                   (local.get $2)
                  )
                 )
                )
                (br_if $block22
                 (i32.and
                  (local.tee $3
                   (i32.load offset=12
                    (local.get $0)
                   )
                  )
                  (i32.const 1)
                 )
                )
                (br_if $block24
                 (i32.eq
                  (i32.shr_u
                   (local.get $3)
                   (i32.const 1)
                  )
                  (local.get $6)
                 )
                )
               )
               (i32.store
                (i32.const 1051472)
                (select
                 (local.tee $0
                  (i32.load
                   (i32.const 1051472)
                  )
                 )
                 (local.get $1)
                 (i32.lt_u
                  (local.get $0)
                  (local.get $1)
                 )
                )
               )
               (local.set $3
                (i32.add
                 (local.get $1)
                 (local.get $4)
                )
               )
               (local.set $0
                (i32.const 1051156)
               )
               (block $block26
                (block $block25
                 (loop $label6
                  (if
                   (i32.ne
                    (local.get $3)
                    (local.tee $7
                     (i32.load
                      (local.get $0)
                     )
                    )
                   )
                   (then
                    (br_if $label6
                     (local.tee $0
                      (i32.load offset=8
                       (local.get $0)
                      )
                     )
                    )
                    (br $block25)
                   )
                  )
                 )
                 (br_if $block25
                  (i32.and
                   (local.tee $3
                    (i32.load offset=12
                     (local.get $0)
                    )
                   )
                   (i32.const 1)
                  )
                 )
                 (br_if $block26
                  (i32.eq
                   (i32.shr_u
                    (local.get $3)
                    (i32.const 1)
                   )
                   (local.get $6)
                  )
                 )
                )
                (local.set $0
                 (i32.const 1051156)
                )
                (loop $label7
                 (block $block27
                  (if
                   (i32.ge_u
                    (local.get $2)
                    (local.tee $3
                     (i32.load
                      (local.get $0)
                     )
                    )
                   )
                   (then
                    (br_if $block27
                     (i32.lt_u
                      (local.get $2)
                      (local.tee $7
                       (i32.add
                        (local.get $3)
                        (i32.load offset=4
                         (local.get $0)
                        )
                       )
                      )
                     )
                    )
                   )
                  )
                  (local.set $0
                   (i32.load offset=8
                    (local.get $0)
                   )
                  )
                  (br $label7)
                 )
                )
                (i32.store
                 (i32.const 1051456)
                 (local.tee $3
                  (i32.sub
                   (local.tee $0
                    (i32.and
                     (i32.add
                      (local.get $1)
                      (i32.const 15)
                     )
                     (i32.const -8)
                    )
                   )
                   (i32.const 8)
                  )
                 )
                )
                (i32.store
                 (i32.const 1051448)
                 (local.tee $0
                  (i32.add
                   (i32.add
                    (local.tee $9
                     (i32.sub
                      (local.get $4)
                      (i32.const 40)
                     )
                    )
                    (i32.sub
                     (local.get $1)
                     (local.get $0)
                    )
                   )
                   (i32.const 8)
                  )
                 )
                )
                (i32.store offset=4
                 (local.get $3)
                 (i32.or
                  (local.get $0)
                  (i32.const 1)
                 )
                )
                (i32.store offset=4
                 (i32.add
                  (local.get $1)
                  (local.get $9)
                 )
                 (i32.const 40)
                )
                (i32.store
                 (i32.const 1051468)
                 (i32.const 2097152)
                )
                (i32.store offset=4
                 (local.tee $3
                  (select
                   (local.get $2)
                   (local.tee $0
                    (i32.sub
                     (i32.and
                      (i32.sub
                       (local.get $7)
                       (i32.const 32)
                      )
                      (i32.const -8)
                     )
                     (i32.const 8)
                    )
                   )
                   (i32.lt_u
                    (local.get $0)
                    (i32.add
                     (local.get $2)
                     (i32.const 16)
                    )
                   )
                  )
                 )
                 (i32.const 27)
                )
                (local.set $10
                 (i64.load align=4
                  (i32.const 1051156)
                 )
                )
                (i64.store align=4
                 (i32.add
                  (local.get $3)
                  (i32.const 16)
                 )
                 (i64.load align=4
                  (i32.const 1051164)
                 )
                )
                (i64.store offset=8 align=4
                 (local.get $3)
                 (local.get $10)
                )
                (i32.store
                 (i32.const 1051168)
                 (local.get $6)
                )
                (i32.store
                 (i32.const 1051160)
                 (local.get $4)
                )
                (i32.store
                 (i32.const 1051156)
                 (local.get $1)
                )
                (i32.store
                 (i32.const 1051164)
                 (i32.add
                  (local.get $3)
                  (i32.const 8)
                 )
                )
                (local.set $0
                 (i32.add
                  (local.get $3)
                  (i32.const 28)
                 )
                )
                (loop $label8
                 (i32.store
                  (local.get $0)
                  (i32.const 7)
                 )
                 (br_if $label8
                  (i32.lt_u
                   (local.tee $0
                    (i32.add
                     (local.get $0)
                     (i32.const 4)
                    )
                   )
                   (local.get $7)
                  )
                 )
                )
                (br_if $block23
                 (i32.eq
                  (local.get $2)
                  (local.get $3)
                 )
                )
                (i32.store offset=4
                 (local.get $3)
                 (i32.and
                  (i32.load offset=4
                   (local.get $3)
                  )
                  (i32.const -2)
                 )
                )
                (i32.store offset=4
                 (local.get $2)
                 (i32.or
                  (local.tee $0
                   (i32.sub
                    (local.get $3)
                    (local.get $2)
                   )
                  )
                  (i32.const 1)
                 )
                )
                (i32.store
                 (local.get $3)
                 (local.get $0)
                )
                (if
                 (i32.ge_u
                  (local.get $0)
                  (i32.const 256)
                 )
                 (then
                  (call $10
                   (local.get $2)
                   (local.get $0)
                  )
                  (br $block23)
                 )
                )
                (local.set $1
                 (i32.add
                  (i32.and
                   (local.get $0)
                   (i32.const 248)
                  )
                  (i32.const 1051172)
                 )
                )
                (local.set $0
                 (block $block28 (result i32)
                  (if
                   (i32.eqz
                    (i32.and
                     (local.tee $3
                      (i32.load
                       (i32.const 1051436)
                      )
                     )
                     (local.tee $0
                      (i32.shl
                       (i32.const 1)
                       (i32.shr_u
                        (local.get $0)
                        (i32.const 3)
                       )
                      )
                     )
                    )
                   )
                   (then
                    (i32.store
                     (i32.const 1051436)
                     (i32.or
                      (local.get $0)
                      (local.get $3)
                     )
                    )
                    (br $block28
                     (local.get $1)
                    )
                   )
                  )
                  (i32.load offset=8
                   (local.get $1)
                  )
                 )
                )
                (i32.store offset=8
                 (local.get $1)
                 (local.get $2)
                )
                (i32.store offset=12
                 (local.get $0)
                 (local.get $2)
                )
                (i32.store offset=12
                 (local.get $2)
                 (local.get $1)
                )
                (i32.store offset=8
                 (local.get $2)
                 (local.get $0)
                )
                (br $block23)
               )
               (i32.store
                (local.get $0)
                (local.get $1)
               )
               (i32.store offset=4
                (local.get $0)
                (i32.add
                 (i32.load offset=4
                  (local.get $0)
                 )
                 (local.get $4)
                )
               )
               (i32.store offset=4
                (local.tee $2
                 (i32.sub
                  (i32.and
                   (i32.add
                    (local.get $1)
                    (i32.const 15)
                   )
                   (i32.const -8)
                  )
                  (i32.const 8)
                 )
                )
                (i32.or
                 (local.get $5)
                 (i32.const 3)
                )
               )
               (local.set $5
                (i32.sub
                 (local.tee $4
                  (i32.sub
                   (i32.and
                    (i32.add
                     (local.get $7)
                     (i32.const 15)
                    )
                    (i32.const -8)
                   )
                   (i32.const 8)
                  )
                 )
                 (local.tee $0
                  (i32.add
                   (local.get $2)
                   (local.get $5)
                  )
                 )
                )
               )
               (br_if $block29
                (i32.eq
                 (local.get $4)
                 (i32.load
                  (i32.const 1051456)
                 )
                )
               )
               (br_if $block30
                (i32.eq
                 (local.get $4)
                 (i32.load
                  (i32.const 1051452)
                 )
                )
               )
               (if
                (i32.eq
                 (i32.and
                  (local.tee $1
                   (i32.load offset=4
                    (local.get $4)
                   )
                  )
                  (i32.const 3)
                 )
                 (i32.const 1)
                )
                (then
                 (call $8
                  (local.get $4)
                  (local.tee $1
                   (i32.and
                    (local.get $1)
                    (i32.const -8)
                   )
                  )
                 )
                 (local.set $5
                  (i32.add
                   (local.get $1)
                   (local.get $5)
                  )
                 )
                 (local.set $1
                  (i32.load offset=4
                   (local.tee $4
                    (i32.add
                     (local.get $1)
                     (local.get $4)
                    )
                   )
                  )
                 )
                )
               )
               (i32.store offset=4
                (local.get $4)
                (i32.and
                 (local.get $1)
                 (i32.const -2)
                )
               )
               (i32.store offset=4
                (local.get $0)
                (i32.or
                 (local.get $5)
                 (i32.const 1)
                )
               )
               (i32.store
                (i32.add
                 (local.get $0)
                 (local.get $5)
                )
                (local.get $5)
               )
               (if
                (i32.ge_u
                 (local.get $5)
                 (i32.const 256)
                )
                (then
                 (call $10
                  (local.get $0)
                  (local.get $5)
                 )
                 (br $block31)
                )
               )
               (local.set $1
                (i32.add
                 (i32.and
                  (local.get $5)
                  (i32.const 248)
                 )
                 (i32.const 1051172)
                )
               )
               (local.set $3
                (block $block32 (result i32)
                 (if
                  (i32.eqz
                   (i32.and
                    (local.tee $3
                     (i32.load
                      (i32.const 1051436)
                     )
                    )
                    (local.tee $4
                     (i32.shl
                      (i32.const 1)
                      (i32.shr_u
                       (local.get $5)
                       (i32.const 3)
                      )
                     )
                    )
                   )
                  )
                  (then
                   (i32.store
                    (i32.const 1051436)
                    (i32.or
                     (local.get $3)
                     (local.get $4)
                    )
                   )
                   (br $block32
                    (local.get $1)
                   )
                  )
                 )
                 (i32.load offset=8
                  (local.get $1)
                 )
                )
               )
               (i32.store offset=8
                (local.get $1)
                (local.get $0)
               )
               (i32.store offset=12
                (local.get $3)
                (local.get $0)
               )
               (i32.store offset=12
                (local.get $0)
                (local.get $1)
               )
               (i32.store offset=8
                (local.get $0)
                (local.get $3)
               )
               (br $block31)
              )
             )
             (i32.store
              (i32.const 1051448)
              (local.tee $1
               (i32.sub
                (local.get $0)
                (local.get $5)
               )
              )
             )
             (i32.store
              (i32.const 1051456)
              (local.tee $2
               (i32.add
                (local.tee $0
                 (i32.load
                  (i32.const 1051456)
                 )
                )
                (local.get $5)
               )
              )
             )
             (i32.store offset=4
              (local.get $2)
              (i32.or
               (local.get $1)
               (i32.const 1)
              )
             )
             (i32.store offset=4
              (local.get $0)
              (i32.or
               (local.get $5)
               (i32.const 3)
              )
             )
             (br $block
              (i32.add
               (local.get $0)
               (i32.const 8)
              )
             )
            )
           )
           (local.set $0
            (i32.load
             (i32.const 1051452)
            )
           )
           (block $block33
            (if
             (i32.le_u
              (local.tee $2
               (i32.sub
                (local.get $1)
                (local.get $5)
               )
              )
              (i32.const 15)
             )
             (then
              (i32.store
               (i32.const 1051452)
               (i32.const 0)
              )
              (i32.store
               (i32.const 1051444)
               (i32.const 0)
              )
              (i32.store offset=4
               (local.get $0)
               (i32.or
                (local.get $1)
                (i32.const 3)
               )
              )
              (i32.store offset=4
               (local.tee $1
                (i32.add
                 (local.get $0)
                 (local.get $1)
                )
               )
               (i32.or
                (i32.load offset=4
                 (local.get $1)
                )
                (i32.const 1)
               )
              )
              (br $block33)
             )
            )
            (i32.store
             (i32.const 1051444)
             (local.get $2)
            )
            (i32.store
             (i32.const 1051452)
             (local.tee $3
              (i32.add
               (local.get $0)
               (local.get $5)
              )
             )
            )
            (i32.store offset=4
             (local.get $3)
             (i32.or
              (local.get $2)
              (i32.const 1)
             )
            )
            (i32.store
             (i32.add
              (local.get $0)
              (local.get $1)
             )
             (local.get $2)
            )
            (i32.store offset=4
             (local.get $0)
             (i32.or
              (local.get $5)
              (i32.const 3)
             )
            )
           )
           (br $block
            (i32.add
             (local.get $0)
             (i32.const 8)
            )
           )
          )
          (i32.store offset=4
           (local.get $0)
           (i32.add
            (local.get $4)
            (local.get $7)
           )
          )
          (i32.store
           (i32.const 1051456)
           (local.tee $2
            (i32.sub
             (local.tee $1
              (i32.and
               (i32.add
                (local.tee $0
                 (i32.load
                  (i32.const 1051456)
                 )
                )
                (i32.const 15)
               )
               (i32.const -8)
              )
             )
             (i32.const 8)
            )
           )
          )
          (i32.store
           (i32.const 1051448)
           (local.tee $1
            (i32.add
             (i32.add
              (local.tee $3
               (i32.add
                (i32.load
                 (i32.const 1051448)
                )
                (local.get $4)
               )
              )
              (i32.sub
               (local.get $0)
               (local.get $1)
              )
             )
             (i32.const 8)
            )
           )
          )
          (i32.store offset=4
           (local.get $2)
           (i32.or
            (local.get $1)
            (i32.const 1)
           )
          )
          (i32.store offset=4
           (i32.add
            (local.get $0)
            (local.get $3)
           )
           (i32.const 40)
          )
          (i32.store
           (i32.const 1051468)
           (i32.const 2097152)
          )
          (br $block23)
         )
         (i32.store
          (i32.const 1051456)
          (local.get $0)
         )
         (i32.store
          (i32.const 1051448)
          (local.tee $1
           (i32.add
            (i32.load
             (i32.const 1051448)
            )
            (local.get $5)
           )
          )
         )
         (i32.store offset=4
          (local.get $0)
          (i32.or
           (local.get $1)
           (i32.const 1)
          )
         )
         (br $block31)
        )
        (i32.store
         (i32.const 1051452)
         (local.get $0)
        )
        (i32.store
         (i32.const 1051444)
         (local.tee $1
          (i32.add
           (i32.load
            (i32.const 1051444)
           )
           (local.get $5)
          )
         )
        )
        (i32.store offset=4
         (local.get $0)
         (i32.or
          (local.get $1)
          (i32.const 1)
         )
        )
        (i32.store
         (i32.add
          (local.get $0)
          (local.get $1)
         )
         (local.get $1)
        )
       )
       (br $block
        (i32.add
         (local.get $2)
         (i32.const 8)
        )
       )
      )
      (drop
       (br_if $block
        (i32.const 0)
        (i32.le_u
         (local.tee $0
          (i32.load
           (i32.const 1051448)
          )
         )
         (local.get $5)
        )
       )
      )
      (i32.store
       (i32.const 1051448)
       (local.tee $1
        (i32.sub
         (local.get $0)
         (local.get $5)
        )
       )
      )
      (i32.store
       (i32.const 1051456)
       (local.tee $2
        (i32.add
         (local.tee $0
          (i32.load
           (i32.const 1051456)
          )
         )
         (local.get $5)
        )
       )
      )
      (i32.store offset=4
       (local.get $2)
       (i32.or
        (local.get $1)
        (i32.const 1)
       )
      )
      (i32.store offset=4
       (local.get $0)
       (i32.or
        (local.get $5)
        (i32.const 3)
       )
      )
      (br $block
       (i32.add
        (local.get $0)
        (i32.const 8)
       )
      )
     )
     (i32.store offset=24
      (local.get $0)
      (local.get $7)
     )
     (if
      (local.tee $1
       (i32.load offset=16
        (local.get $2)
       )
      )
      (then
       (i32.store offset=16
        (local.get $0)
        (local.get $1)
       )
       (i32.store offset=24
        (local.get $1)
        (local.get $0)
       )
      )
     )
     (br_if $block19
      (i32.eqz
       (local.tee $1
        (i32.load offset=20
         (local.get $2)
        )
       )
      )
     )
     (i32.store offset=20
      (local.get $0)
      (local.get $1)
     )
     (i32.store offset=24
      (local.get $1)
      (local.get $0)
     )
    )
    (block $block34
     (if
      (i32.ge_u
       (local.get $4)
       (i32.const 16)
      )
      (then
       (i32.store offset=4
        (local.get $2)
        (i32.or
         (local.get $5)
         (i32.const 3)
        )
       )
       (i32.store offset=4
        (local.tee $0
         (i32.add
          (local.get $2)
          (local.get $5)
         )
        )
        (i32.or
         (local.get $4)
         (i32.const 1)
        )
       )
       (i32.store
        (i32.add
         (local.get $0)
         (local.get $4)
        )
        (local.get $4)
       )
       (if
        (i32.ge_u
         (local.get $4)
         (i32.const 256)
        )
        (then
         (call $10
          (local.get $0)
          (local.get $4)
         )
         (br $block34)
        )
       )
       (local.set $1
        (i32.add
         (i32.and
          (local.get $4)
          (i32.const 248)
         )
         (i32.const 1051172)
        )
       )
       (local.set $3
        (block $block35 (result i32)
         (if
          (i32.eqz
           (i32.and
            (local.tee $3
             (i32.load
              (i32.const 1051436)
             )
            )
            (local.tee $4
             (i32.shl
              (i32.const 1)
              (i32.shr_u
               (local.get $4)
               (i32.const 3)
              )
             )
            )
           )
          )
          (then
           (i32.store
            (i32.const 1051436)
            (i32.or
             (local.get $3)
             (local.get $4)
            )
           )
           (br $block35
            (local.get $1)
           )
          )
         )
         (i32.load offset=8
          (local.get $1)
         )
        )
       )
       (i32.store offset=8
        (local.get $1)
        (local.get $0)
       )
       (i32.store offset=12
        (local.get $3)
        (local.get $0)
       )
       (i32.store offset=12
        (local.get $0)
        (local.get $1)
       )
       (i32.store offset=8
        (local.get $0)
        (local.get $3)
       )
       (br $block34)
      )
     )
     (i32.store offset=4
      (local.get $2)
      (i32.or
       (local.tee $0
        (i32.add
         (local.get $4)
         (local.get $5)
        )
       )
       (i32.const 3)
      )
     )
     (i32.store offset=4
      (local.tee $0
       (i32.add
        (local.get $0)
        (local.get $2)
       )
      )
      (i32.or
       (i32.load offset=4
        (local.get $0)
       )
       (i32.const 1)
      )
     )
    )
    (i32.add
     (local.get $2)
     (i32.const 8)
    )
   )
  )
  (global.set $global$0
   (i32.add
    (local.get $8)
    (i32.const 16)
   )
  )
  (local.get $scratch)
 )
 (func $1 (param $0 i32) (param $1 f64)
  (local $2 i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i32)
  (local $6 i32)
  (local $7 i32)
  (local $8 i32)
  (local $9 i32)
  (local $10 i32)
  (local $11 i32)
  (local $12 i32)
  (local $13 i32)
  (local $14 i32)
  (local $15 i32)
  (local $16 i32)
  (local $17 i32)
  (local $18 i32)
  (local $19 i32)
  (local $20 i32)
  (local $21 i32)
  (local $22 i32)
  (local $23 i32)
  (local $24 i32)
  (local $25 i32)
  (local $26 i32)
  (local $27 i32)
  (local $28 i32)
  (local $29 i32)
  (local $30 i32)
  (local $31 i32)
  (local $32 i32)
  (local $33 f64)
  (local $34 f64)
  (local $35 f64)
  (local $36 f64)
  (local $37 i64)
  (local $scratch i32)
  (local $scratch_39 i32)
  (local $scratch_40 i32)
  (local $scratch_41 i32)
  (global.set $global$0
   (local.tee $10
    (i32.sub
     (global.get $global$0)
     (i32.const 48)
    )
   )
  )
  (block $block5
   (block $block45
    (block $block46
     (block $block47
      (block $block
       (if
        (i32.ge_u
         (local.tee $4
          (i32.and
           (local.tee $5
            (i32.wrap_i64
             (i64.shr_u
              (local.tee $37
               (i64.reinterpret_f64
                (local.get $1)
               )
              )
              (i64.const 32)
             )
            )
           )
           (i32.const 2147483647)
          )
         )
         (i32.const 1074752123)
        )
        (then
         (if
          (i32.ge_u
           (local.get $4)
           (i32.const 1075594812)
          )
          (then
           (f64.store
            (local.get $10)
            (local.tee $33
             (f64.convert_i32_s
              (select
               (select
                (i32.const 2147483647)
                (select
                 (block $block2 (result i32)
                  (block $block1
                   (if
                    (i32.ge_u
                     (local.get $4)
                     (i32.const 1094263291)
                    )
                    (then
                     (br_if $block
                      (i32.gt_u
                       (local.get $4)
                       (i32.const 2146435071)
                      )
                     )
                     (local.set $5
                      (f64.ge
                       (local.tee $1
                        (f64.reinterpret_i64
                         (i64.or
                          (i64.and
                           (local.get $37)
                           (i64.const 4503599627370495)
                          )
                          (i64.const 4710765210229538816)
                         )
                        )
                       )
                       (f64.const -2147483648)
                      )
                     )
                     (br_if $block1
                      (i32.eqz
                       (f64.lt
                        (f64.abs
                         (local.get $1)
                        )
                        (f64.const 2147483648)
                       )
                      )
                     )
                     (br $block2
                      (i32.trunc_f64_s
                       (local.get $1)
                      )
                     )
                    )
                   )
                   (block $block3
                    (br_if $block3
                     (i32.lt_s
                      (i32.sub
                       (local.tee $4
                        (i32.shr_u
                         (local.get $4)
                         (i32.const 20)
                        )
                       )
                       (i32.and
                        (i32.wrap_i64
                         (i64.shr_u
                          (i64.reinterpret_f64
                           (local.tee $35
                            (f64.sub
                             (local.tee $1
                              (f64.add
                               (local.get $1)
                               (f64.mul
                                (local.tee $34
                                 (f64.add
                                  (f64.add
                                   (f64.mul
                                    (local.get $1)
                                    (f64.const 0.6366197723675814)
                                   )
                                   (f64.const 6755399441055744)
                                  )
                                  (f64.const -6755399441055744)
                                 )
                                )
                                (f64.const -1.5707963267341256)
                               )
                              )
                             )
                             (local.tee $36
                              (f64.mul
                               (local.get $34)
                               (f64.const 6.077100506506192e-11)
                              )
                             )
                            )
                           )
                          )
                          (i64.const 52)
                         )
                        )
                        (i32.const 2047)
                       )
                      )
                      (i32.const 17)
                     )
                    )
                    (if
                     (i32.lt_s
                      (i32.sub
                       (local.get $4)
                       (i32.and
                        (i32.wrap_i64
                         (i64.shr_u
                          (i64.reinterpret_f64
                           (local.tee $35
                            (f64.sub
                             (local.tee $33
                              (f64.sub
                               (local.get $1)
                               (local.tee $35
                                (f64.mul
                                 (local.get $34)
                                 (f64.const 6.077100506303966e-11)
                                )
                               )
                              )
                             )
                             (local.tee $36
                              (f64.sub
                               (f64.mul
                                (local.get $34)
                                (f64.const 2.0222662487959506e-21)
                               )
                               (f64.sub
                                (f64.sub
                                 (local.get $1)
                                 (local.get $33)
                                )
                                (local.get $35)
                               )
                              )
                             )
                            )
                           )
                          )
                          (i64.const 52)
                         )
                        )
                        (i32.const 2047)
                       )
                      )
                      (i32.const 50)
                     )
                     (then
                      (local.set $1
                       (local.get $33)
                      )
                      (br $block3)
                     )
                    )
                    (local.set $35
                     (f64.sub
                      (local.tee $1
                       (f64.sub
                        (local.get $33)
                        (local.tee $35
                         (f64.mul
                          (local.get $34)
                          (f64.const 2.0222662487111665e-21)
                         )
                        )
                       )
                      )
                      (local.tee $36
                       (f64.sub
                        (f64.mul
                         (local.get $34)
                         (f64.const 8.4784276603689e-32)
                        )
                        (f64.sub
                         (f64.sub
                          (local.get $33)
                          (local.get $1)
                         )
                         (local.get $35)
                        )
                       )
                      )
                     )
                    )
                   )
                   (f64.store
                    (local.get $0)
                    (local.get $35)
                   )
                   (f64.store offset=16
                    (local.get $0)
                    (f64.sub
                     (f64.sub
                      (local.get $1)
                      (local.get $35)
                     )
                     (local.get $36)
                    )
                   )
                   (local.set $4
                    (f64.ge
                     (local.get $34)
                     (f64.const -2147483648)
                    )
                   )
                   (i32.store offset=8
                    (local.get $0)
                    (select
                     (select
                      (i32.const 2147483647)
                      (select
                       (block $block4 (result i32)
                        (if
                         (f64.lt
                          (f64.abs
                           (local.get $34)
                          )
                          (f64.const 2147483648)
                         )
                         (then
                          (br $block4
                           (i32.trunc_f64_s
                            (local.get $34)
                           )
                          )
                         )
                        )
                        (i32.const -2147483648)
                       )
                       (i32.const -2147483648)
                       (local.get $4)
                      )
                      (f64.gt
                       (local.get $34)
                       (f64.const 2147483647)
                      )
                     )
                     (i32.const 0)
                     (f64.eq
                      (local.get $34)
                      (local.get $34)
                     )
                    )
                   )
                   (br $block5)
                  )
                  (i32.const -2147483648)
                 )
                 (i32.const -2147483648)
                 (local.get $5)
                )
                (f64.gt
                 (local.get $1)
                 (f64.const 2147483647)
                )
               )
               (i32.const 0)
               (f64.eq
                (local.get $1)
                (local.get $1)
               )
              )
             )
            )
           )
           (local.set $5
            (f64.ge
             (local.tee $1
              (f64.mul
               (f64.sub
                (local.get $1)
                (local.get $33)
               )
               (f64.const 16777216)
              )
             )
             (f64.const -2147483648)
            )
           )
           (f64.store offset=8
            (local.get $10)
            (local.tee $33
             (f64.convert_i32_s
              (local.tee $5
               (select
                (select
                 (i32.const 2147483647)
                 (select
                  (block $block6 (result i32)
                   (if
                    (f64.lt
                     (f64.abs
                      (local.get $1)
                     )
                     (f64.const 2147483648)
                    )
                    (then
                     (br $block6
                      (i32.trunc_f64_s
                       (local.get $1)
                      )
                     )
                    )
                   )
                   (i32.const -2147483648)
                  )
                  (i32.const -2147483648)
                  (local.get $5)
                 )
                 (f64.gt
                  (local.get $1)
                  (f64.const 2147483647)
                 )
                )
                (i32.const 0)
                (f64.eq
                 (local.get $1)
                 (local.get $1)
                )
               )
              )
             )
            )
           )
           (f64.store offset=16
            (local.get $10)
            (local.tee $1
             (f64.mul
              (f64.sub
               (local.get $1)
               (local.get $33)
              )
              (f64.const 16777216)
             )
            )
           )
           (i64.store offset=40
            (local.get $10)
            (i64.const 0)
           )
           (i64.store offset=32
            (local.get $10)
            (i64.const 0)
           )
           (i64.store offset=24
            (local.get $10)
            (i64.const 0)
           )
           (local.set $17
            (i32.add
             (local.get $10)
             (i32.const 24)
            )
           )
           (global.set $global$0
            (local.tee $3
             (i32.sub
              (global.get $global$0)
              (i32.const 560)
             )
            )
           )
           (i64.store offset=152
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=144
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=136
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=128
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=120
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=112
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=104
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=96
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=88
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=80
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=72
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=64
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=56
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=48
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=40
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=32
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=24
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=16
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=8
            (local.get $3)
            (i64.const 0)
           )
           (i64.store
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=312
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=304
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=296
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=288
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=280
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=272
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=264
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=256
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=248
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=240
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=232
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=224
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=216
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=208
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=200
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=192
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=184
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=176
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=168
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=160
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=472
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=464
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=456
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=448
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=440
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=432
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=424
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=416
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=408
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=400
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=392
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=384
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=376
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=368
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=360
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=352
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=344
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=336
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=328
            (local.get $3)
            (i64.const 0)
           )
           (i64.store offset=320
            (local.get $3)
            (i64.const 0)
           )
           (call $6
            (i32.add
             (local.get $3)
             (i32.const 480)
            )
            (i32.const 80)
           )
           (local.set $9
            (i32.add
             (local.tee $11
              (i32.load
               (i32.const 1050624)
              )
             )
             (local.tee $12
              (i32.sub
               (local.tee $2
                (select
                 (i32.const 3)
                 (select
                  (i32.const 2)
                  (i32.const 1)
                  (local.get $5)
                 )
                 (f64.ne
                  (local.get $1)
                  (f64.const 0)
                 )
                )
               )
               (i32.const 1)
              )
             )
            )
           )
           (local.set $5
            (i32.sub
             (local.tee $14
              (select
               (local.tee $4
                (i32.div_s
                 (i32.sub
                  (local.tee $18
                   (i32.sub
                    (i32.shr_u
                     (local.get $4)
                     (i32.const 20)
                    )
                    (i32.const 1046)
                   )
                  )
                  (i32.const 3)
                 )
                 (i32.const 24)
                )
               )
               (i32.const 0)
               (i32.gt_s
                (local.get $4)
                (i32.const 0)
               )
              )
             )
             (local.get $12)
            )
           )
           (local.set $7
            (i32.add
             (i32.sub
              (i32.shl
               (local.get $14)
               (i32.const 2)
              )
              (i32.shl
               (local.get $2)
               (i32.const 2)
              )
             )
             (i32.const 1050640)
            )
           )
           (local.set $2
            (i32.const 0)
           )
           (loop $label
            (f64.store
             (i32.add
              (local.get $3)
              (i32.shl
               (local.get $2)
               (i32.const 3)
              )
             )
             (if (result f64)
              (i32.lt_s
               (local.get $5)
               (i32.const 0)
              )
              (then
               (f64.const 0)
              )
              (else
               (f64.convert_i32_s
                (i32.load
                 (local.get $7)
                )
               )
              )
             )
            )
            (if
             (local.tee $4
              (i32.lt_u
               (local.get $2)
               (local.get $9)
              )
             )
             (then
              (local.set $7
               (i32.add
                (local.get $7)
                (i32.const 4)
               )
              )
              (local.set $5
               (i32.add
                (local.get $5)
                (i32.const 1)
               )
              )
              (br_if $label
               (i32.le_u
                (local.tee $2
                 (i32.add
                  (local.get $2)
                  (local.get $4)
                 )
                )
                (local.get $9)
               )
              )
             )
            )
           )
           (local.set $4
            (i32.sub
             (local.get $18)
             (i32.const 24)
            )
           )
           (local.set $5
            (i32.const 0)
           )
           (loop $label2
            (local.set $9
             (i32.add
              (local.get $5)
              (local.get $12)
             )
            )
            (local.set $1
             (f64.const 0)
            )
            (local.set $2
             (i32.const 0)
            )
            (loop $label1
             (block $block7
              (local.set $1
               (f64.add
                (local.get $1)
                (f64.mul
                 (f64.load
                  (i32.add
                   (local.get $10)
                   (i32.shl
                    (local.get $2)
                    (i32.const 3)
                   )
                  )
                 )
                 (f64.load
                  (i32.add
                   (local.get $3)
                   (i32.shl
                    (i32.sub
                     (local.get $9)
                     (local.get $2)
                    )
                    (i32.const 3)
                   )
                  )
                 )
                )
               )
              )
              (br_if $block7
               (i32.ge_u
                (local.get $2)
                (local.get $12)
               )
              )
              (br_if $label1
               (i32.le_u
                (local.tee $2
                 (i32.add
                  (local.get $2)
                  (i32.lt_u
                   (local.get $2)
                   (local.get $12)
                  )
                 )
                )
                (local.get $12)
               )
              )
             )
            )
            (f64.store
             (i32.add
              (i32.add
               (local.get $3)
               (i32.const 320)
              )
              (i32.shl
               (local.get $5)
               (i32.const 3)
              )
             )
             (local.get $1)
            )
            (if
             (local.tee $2
              (i32.lt_u
               (local.get $5)
               (local.get $11)
              )
             )
             (then
              (br_if $label2
               (i32.le_u
                (local.tee $5
                 (i32.add
                  (local.get $2)
                  (local.get $5)
                 )
                )
                (local.get $11)
               )
              )
             )
            )
           )
           (local.set $35
            (f64.mul
             (select
              (select
               (f64.const inf)
               (f64.const 8988465674311579538646525e283)
               (local.tee $19
                (i32.gt_u
                 (local.tee $6
                  (i32.add
                   (local.get $4)
                   (local.tee $24
                    (i32.mul
                     (local.get $14)
                     (i32.const -24)
                    )
                   )
                  )
                 )
                 (i32.const 2046)
                )
               )
              )
              (select
               (select
                (f64.const 0)
                (f64.const 2.004168360008973e-292)
                (local.tee $20
                 (i32.lt_u
                  (local.get $6)
                  (i32.const -1991)
                 )
                )
               )
               (f64.const 1)
               (local.tee $21
                (i32.lt_s
                 (local.get $6)
                 (i32.const -1022)
                )
               )
              )
              (local.tee $22
               (i32.gt_s
                (local.get $6)
                (i32.const 1023)
               )
              )
             )
             (f64.reinterpret_i64
              (i64.shl
               (i64.extend_i32_u
                (i32.add
                 (select
                  (local.tee $25
                   (select
                    (i32.sub
                     (select
                      (i32.const 3069)
                      (local.get $6)
                      (i32.ge_u
                       (local.get $6)
                       (i32.const 3069)
                      )
                     )
                     (i32.const 2046)
                    )
                    (i32.sub
                     (local.get $6)
                     (i32.const 1023)
                    )
                    (local.get $19)
                   )
                  )
                  (select
                   (local.tee $26
                    (select
                     (i32.add
                      (select
                       (i32.const -2960)
                       (local.get $6)
                       (i32.le_u
                        (local.get $6)
                        (i32.const -2960)
                       )
                      )
                      (i32.const 1938)
                     )
                     (i32.add
                      (local.get $6)
                      (i32.const 969)
                     )
                     (local.get $20)
                    )
                   )
                   (local.get $6)
                   (local.get $21)
                  )
                  (local.get $22)
                 )
                 (i32.const 1023)
                )
               )
               (i64.const 52)
              )
             )
            )
           )
           (local.set $9
            (i32.add
             (local.tee $15
              (i32.add
               (local.get $3)
               (i32.const 476)
              )
             )
             (i32.shl
              (local.get $11)
              (i32.const 2)
             )
            )
           )
           (local.set $27
            (i32.and
             (i32.sub
              (i32.const 23)
              (local.get $6)
             )
             (i32.const 31)
            )
           )
           (local.set $23
            (i32.and
             (i32.sub
              (i32.const 24)
              (local.get $6)
             )
             (i32.const 31)
            )
           )
           (local.set $28
            (i32.add
             (local.get $3)
             (i32.const 312)
            )
           )
           (local.set $29
            (i32.sub
             (local.get $6)
             (i32.const 1)
            )
           )
           (local.set $5
            (local.get $11)
           )
           (block $block29
            (loop $label8
             (local.set $1
              (f64.load
               (i32.add
                (i32.add
                 (local.get $3)
                 (i32.const 320)
                )
                (i32.shl
                 (local.tee $4
                  (local.get $5)
                 )
                 (i32.const 3)
                )
               )
              )
             )
             (block $block8
              (br_if $block8
               (i32.eqz
                (local.get $4)
               )
              )
              (local.set $8
               (i32.add
                (local.get $3)
                (i32.const 480)
               )
              )
              (local.set $2
               (local.get $4)
              )
              (loop $label3
               (local.set $5
                (f64.ge
                 (local.tee $33
                  (f64.mul
                   (local.get $1)
                   (f64.const 5.9604644775390625e-08)
                  )
                 )
                 (f64.const -2147483648)
                )
               )
               (local.set $5
                (f64.ge
                 (local.tee $1
                  (f64.add
                   (local.get $1)
                   (f64.mul
                    (local.tee $33
                     (f64.convert_i32_s
                      (select
                       (select
                        (i32.const 2147483647)
                        (select
                         (if (result i32)
                          (f64.lt
                           (f64.abs
                            (local.get $33)
                           )
                           (f64.const 2147483648)
                          )
                          (then
                           (i32.trunc_f64_s
                            (local.get $33)
                           )
                          )
                          (else
                           (i32.const -2147483648)
                          )
                         )
                         (i32.const -2147483648)
                         (local.get $5)
                        )
                        (f64.gt
                         (local.get $33)
                         (f64.const 2147483647)
                        )
                       )
                       (i32.const 0)
                       (f64.eq
                        (local.get $33)
                        (local.get $33)
                       )
                      )
                     )
                    )
                    (f64.const -16777216)
                   )
                  )
                 )
                 (f64.const -2147483648)
                )
               )
               (i32.store
                (local.get $8)
                (select
                 (select
                  (i32.const 2147483647)
                  (select
                   (block $block9 (result i32)
                    (if
                     (f64.lt
                      (f64.abs
                       (local.get $1)
                      )
                      (f64.const 2147483648)
                     )
                     (then
                      (br $block9
                       (i32.trunc_f64_s
                        (local.get $1)
                       )
                      )
                     )
                    )
                    (i32.const -2147483648)
                   )
                   (i32.const -2147483648)
                   (local.get $5)
                  )
                  (f64.gt
                   (local.get $1)
                   (f64.const 2147483647)
                  )
                 )
                 (i32.const 0)
                 (f64.eq
                  (local.get $1)
                  (local.get $1)
                 )
                )
               )
               (local.set $1
                (f64.add
                 (f64.load
                  (i32.add
                   (local.get $28)
                   (i32.shl
                    (local.get $2)
                    (i32.const 3)
                   )
                  )
                 )
                 (local.get $33)
                )
               )
               (br_if $block8
                (local.tee $5
                 (i32.lt_u
                  (local.get $2)
                  (i32.const 2)
                 )
                )
               )
               (local.set $8
                (i32.add
                 (local.get $8)
                 (i32.const 4)
                )
               )
               (br_if $label3
                (local.tee $2
                 (select
                  (i32.const 1)
                  (i32.sub
                   (local.get $2)
                   (i32.const 1)
                  )
                  (local.get $5)
                 )
                )
               )
              )
             )
             (local.set $5
              (block $block11 (result i32)
               (block $block10
                (if
                 (i32.eqz
                  (local.get $22)
                 )
                 (then
                  (br_if $block10
                   (local.get $21)
                  )
                  (br $block11
                   (local.get $6)
                  )
                 )
                )
                (local.set $1
                 (select
                  (f64.mul
                   (local.tee $1
                    (f64.mul
                     (local.get $1)
                     (f64.const 8988465674311579538646525e283)
                    )
                   )
                   (f64.const 8988465674311579538646525e283)
                  )
                  (local.get $1)
                  (local.get $19)
                 )
                )
                (br $block11
                 (local.get $25)
                )
               )
               (local.set $1
                (select
                 (f64.mul
                  (local.tee $1
                   (f64.mul
                    (local.get $1)
                    (f64.const 2.004168360008973e-292)
                   )
                  )
                  (f64.const 2.004168360008973e-292)
                 )
                 (local.get $1)
                 (local.get $20)
                )
               )
               (local.get $26)
              )
             )
             (local.set $5
              (f64.ge
               (local.tee $1
                (f64.add
                 (local.tee $1
                  (f64.mul
                   (local.get $1)
                   (f64.reinterpret_i64
                    (i64.shl
                     (i64.extend_i32_u
                      (i32.add
                       (local.get $5)
                       (i32.const 1023)
                      )
                     )
                     (i64.const 52)
                    )
                   )
                  )
                 )
                 (f64.mul
                  (f64.floor
                   (f64.mul
                    (local.get $1)
                    (f64.const 0.125)
                   )
                  )
                  (f64.const -8)
                 )
                )
               )
               (f64.const -2147483648)
              )
             )
             (local.set $1
              (f64.sub
               (local.get $1)
               (f64.convert_i32_s
                (local.tee $16
                 (select
                  (select
                   (i32.const 2147483647)
                   (select
                    (block $block12 (result i32)
                     (if
                      (f64.lt
                       (f64.abs
                        (local.get $1)
                       )
                       (f64.const 2147483648)
                      )
                      (then
                       (br $block12
                        (i32.trunc_f64_s
                         (local.get $1)
                        )
                       )
                      )
                     )
                     (i32.const -2147483648)
                    )
                    (i32.const -2147483648)
                    (local.get $5)
                   )
                   (f64.gt
                    (local.get $1)
                    (f64.const 2147483647)
                   )
                  )
                  (i32.const 0)
                  (f64.eq
                   (local.get $1)
                   (local.get $1)
                  )
                 )
                )
               )
              )
             )
             (local.set $13
              (block $block14 (result i32)
               (block $block18
                (block $block17
                 (block $block16
                  (block $block15
                   (br_if $block16
                    (i32.le_s
                     (local.tee $5
                      (block $block13 (result i32)
                       (if
                        (i32.eqz
                         (local.tee $30
                          (i32.gt_s
                           (local.get $6)
                           (i32.const 0)
                          )
                         )
                        )
                        (then
                         (if
                          (i32.eqz
                           (local.get $6)
                          )
                          (then
                           (br $block13
                            (i32.shr_s
                             (i32.load
                              (i32.add
                               (local.get $15)
                               (i32.shl
                                (local.get $4)
                                (i32.const 2)
                               )
                              )
                             )
                             (i32.const 23)
                            )
                           )
                          )
                         )
                         (local.set $5
                          (i32.const 2)
                         )
                         (drop
                          (br_if $block14
                           (i32.const 0)
                           (i32.eqz
                            (f64.ge
                             (local.get $1)
                             (f64.const 0.5)
                            )
                           )
                          )
                         )
                         (br $block15)
                        )
                       )
                       (i32.store
                        (local.tee $5
                         (i32.add
                          (local.get $15)
                          (i32.shl
                           (local.get $4)
                           (i32.const 2)
                          )
                         )
                        )
                        (local.tee $2
                         (i32.sub
                          (local.tee $5
                           (i32.load
                            (local.get $5)
                           )
                          )
                          (i32.shl
                           (local.tee $5
                            (i32.shr_s
                             (local.get $5)
                             (local.get $23)
                            )
                           )
                           (local.get $23)
                          )
                         )
                        )
                       )
                       (local.set $16
                        (i32.add
                         (local.get $5)
                         (local.get $16)
                        )
                       )
                       (i32.shr_s
                        (local.get $2)
                        (local.get $27)
                       )
                      )
                     )
                     (i32.const 0)
                    )
                   )
                  )
                  (br_if $block17
                   (local.get $4)
                  )
                  (local.set $8
                   (i32.const 0)
                  )
                  (br $block18)
                 )
                 (br $block14
                  (local.get $5)
                 )
                )
                (local.set $7
                 (i32.const 0)
                )
                (local.set $8
                 (i32.const 0)
                )
                (if
                 (i32.ne
                  (local.get $4)
                  (i32.const 1)
                 )
                 (then
                  (local.set $31
                   (i32.and
                    (local.get $4)
                    (i32.const 30)
                   )
                  )
                  (local.set $2
                   (i32.add
                    (local.get $3)
                    (i32.const 480)
                   )
                  )
                  (loop $label4
                   (local.set $13
                    (i32.load
                     (local.get $2)
                    )
                   )
                   (local.set $13
                    (block $block20 (result i32)
                     (block $block19
                      (i32.store
                       (local.get $2)
                       (i32.sub
                        (if (result i32)
                         (local.get $8)
                         (then
                          (i32.const 16777215)
                         )
                         (else
                          (br_if $block19
                           (i32.eqz
                            (local.get $13)
                           )
                          )
                          (i32.const 16777216)
                         )
                        )
                        (local.get $13)
                       )
                      )
                      (br $block20
                       (i32.const 0)
                      )
                     )
                     (i32.const 1)
                    )
                   )
                   (local.set $8
                    (i32.load
                     (local.tee $32
                      (i32.add
                       (local.get $2)
                       (i32.const 4)
                      )
                     )
                    )
                   )
                   (local.set $8
                    (block $block22 (result i32)
                     (block $block21
                      (i32.store
                       (local.get $32)
                       (i32.sub
                        (if (result i32)
                         (local.get $13)
                         (then
                          (br_if $block21
                           (i32.eqz
                            (local.get $8)
                           )
                          )
                          (i32.const 16777216)
                         )
                         (else
                          (i32.const 16777215)
                         )
                        )
                        (local.get $8)
                       )
                      )
                      (br $block22
                       (i32.const 1)
                      )
                     )
                     (i32.const 0)
                    )
                   )
                   (local.set $2
                    (i32.add
                     (local.get $2)
                     (i32.const 8)
                    )
                   )
                   (br_if $label4
                    (i32.ne
                     (local.get $31)
                     (local.tee $7
                      (i32.add
                       (local.get $7)
                       (i32.const 2)
                      )
                     )
                    )
                   )
                  )
                 )
                )
                (br_if $block18
                 (i32.eqz
                  (i32.and
                   (local.get $4)
                   (i32.const 1)
                  )
                 )
                )
                (local.set $2
                 (i32.load
                  (local.tee $7
                   (i32.add
                    (i32.add
                     (local.get $3)
                     (i32.const 480)
                    )
                    (i32.shl
                     (local.get $7)
                     (i32.const 2)
                    )
                   )
                  )
                 )
                )
                (block $block23
                 (i32.store
                  (local.get $7)
                  (i32.sub
                   (if (result i32)
                    (local.get $8)
                    (then
                     (i32.const 16777215)
                    )
                    (else
                     (br_if $block23
                      (i32.eqz
                       (local.get $2)
                      )
                     )
                     (i32.const 16777216)
                    )
                   )
                   (local.get $2)
                  )
                 )
                 (local.set $8
                  (i32.const 1)
                 )
                 (br $block18)
                )
                (local.set $8
                 (i32.const 0)
                )
               )
               (block $block24
                (br_if $block24
                 (i32.eqz
                  (local.get $30)
                 )
                )
                (local.set $2
                 (i32.const 8388607)
                )
                (block $block25
                 (block $block26
                  (br_table $block25 $block26 $block24
                   (local.get $29)
                  )
                 )
                 (local.set $2
                  (i32.const 4194303)
                 )
                )
                (i32.store
                 (local.tee $7
                  (i32.add
                   (local.get $15)
                   (i32.shl
                    (local.get $4)
                    (i32.const 2)
                   )
                  )
                 )
                 (i32.and
                  (i32.load
                   (local.get $7)
                  )
                  (local.get $2)
                 )
                )
               )
               (local.set $16
                (i32.add
                 (local.get $16)
                 (i32.const 1)
                )
               )
               (drop
                (br_if $block14
                 (local.get $5)
                 (i32.ne
                  (local.get $5)
                  (i32.const 2)
                 )
                )
               )
               (local.set $1
                (select
                 (f64.sub
                  (local.tee $1
                   (f64.sub
                    (f64.const 1)
                    (local.get $1)
                   )
                  )
                  (local.get $35)
                 )
                 (local.get $1)
                 (local.get $8)
                )
               )
               (i32.const 2)
              )
             )
             (if
              (f64.eq
               (local.get $1)
               (f64.const 0)
              )
              (then
               (local.set $2
                (local.get $9)
               )
               (local.set $5
                (local.get $4)
               )
               (block $block27
                (br_if $block27
                 (i32.gt_u
                  (local.get $11)
                  (local.tee $8
                   (i32.sub
                    (local.get $4)
                    (i32.const 1)
                   )
                  )
                 )
                )
                (local.set $7
                 (i32.const 0)
                )
                (loop $label5
                 (block $block28
                  (local.set $7
                   (i32.or
                    (i32.load
                     (i32.add
                      (i32.add
                       (local.get $3)
                       (i32.const 480)
                      )
                      (i32.shl
                       (local.get $8)
                       (i32.const 2)
                      )
                     )
                    )
                    (local.get $7)
                   )
                  )
                  (br_if $block28
                   (i32.le_u
                    (local.get $8)
                    (local.get $11)
                   )
                  )
                  (br_if $label5
                   (i32.le_u
                    (local.get $11)
                    (local.tee $8
                     (i32.sub
                      (local.get $8)
                      (i32.gt_u
                       (local.get $8)
                       (local.get $11)
                      )
                     )
                    )
                   )
                  )
                 )
                )
                (local.set $5
                 (local.get $4)
                )
                (br_if $block27
                 (i32.eqz
                  (local.get $7)
                 )
                )
                (local.set $2
                 (i32.add
                  (i32.add
                   (i32.shl
                    (local.get $4)
                    (i32.const 2)
                   )
                   (local.get $3)
                  )
                  (i32.const 476)
                 )
                )
                (loop $label6
                 (local.set $4
                  (i32.sub
                   (local.get $4)
                   (i32.const 1)
                  )
                 )
                 (local.set $6
                  (i32.sub
                   (local.get $6)
                   (i32.const 24)
                  )
                 )
                 (br_if $label6
                  (i32.eqz
                   (block (result i32)
                    (local.set $scratch
                     (i32.load
                      (local.get $2)
                     )
                    )
                    (local.set $2
                     (i32.sub
                      (local.get $2)
                      (i32.const 4)
                     )
                    )
                    (local.get $scratch)
                   )
                  )
                 )
                )
                (br $block29)
               )
               (loop $label7
                (local.set $5
                 (i32.add
                  (local.get $5)
                  (i32.const 1)
                 )
                )
                (br_if $label7
                 (i32.eqz
                  (block (result i32)
                   (local.set $scratch_39
                    (i32.load
                     (local.get $2)
                    )
                   )
                   (local.set $2
                    (i32.sub
                     (local.get $2)
                     (i32.const 4)
                    )
                   )
                   (local.get $scratch_39)
                  )
                 )
                )
               )
               (br_if $label8
                (i32.ge_u
                 (local.get $4)
                 (local.get $5)
                )
               )
               (local.set $7
                (i32.add
                 (local.get $4)
                 (i32.const 1)
                )
               )
               (loop $label10
                (f64.store
                 (i32.add
                  (local.get $3)
                  (i32.shl
                   (local.tee $4
                    (i32.add
                     (local.get $7)
                     (local.get $12)
                    )
                   )
                   (i32.const 3)
                  )
                 )
                 (f64.convert_i32_s
                  (i32.load
                   (i32.add
                    (i32.shl
                     (i32.add
                      (local.get $7)
                      (local.get $14)
                     )
                     (i32.const 2)
                    )
                    (i32.const 1050636)
                   )
                  )
                 )
                )
                (local.set $2
                 (i32.const 0)
                )
                (local.set $1
                 (f64.const 0)
                )
                (loop $label9
                 (block $block30
                  (local.set $1
                   (f64.add
                    (local.get $1)
                    (f64.mul
                     (f64.load
                      (i32.add
                       (local.get $10)
                       (i32.shl
                        (local.get $2)
                        (i32.const 3)
                       )
                      )
                     )
                     (f64.load
                      (i32.add
                       (local.get $3)
                       (i32.shl
                        (i32.sub
                         (local.get $4)
                         (local.get $2)
                        )
                        (i32.const 3)
                       )
                      )
                     )
                    )
                   )
                  )
                  (br_if $block30
                   (i32.ge_u
                    (local.get $2)
                    (local.get $12)
                   )
                  )
                  (br_if $label9
                   (i32.le_u
                    (local.tee $2
                     (i32.add
                      (local.get $2)
                      (i32.lt_u
                       (local.get $2)
                       (local.get $12)
                      )
                     )
                    )
                    (local.get $12)
                   )
                  )
                 )
                )
                (f64.store
                 (i32.add
                  (i32.add
                   (local.get $3)
                   (i32.const 320)
                  )
                  (i32.shl
                   (local.get $7)
                   (i32.const 3)
                  )
                 )
                 (local.get $1)
                )
                (br_if $label8
                 (i32.le_u
                  (local.get $5)
                  (local.get $7)
                 )
                )
                (local.set $7
                 (local.tee $4
                  (i32.add
                   (local.get $7)
                   (i32.gt_u
                    (local.get $5)
                    (local.get $7)
                   )
                  )
                 )
                )
                (br_if $label10
                 (i32.le_u
                  (local.get $4)
                  (local.get $5)
                 )
                )
               )
               (br $label8)
              )
             )
            )
            (block $block31
             (block $block33
              (block $block32
               (if
                (i32.le_s
                 (local.tee $2
                  (i32.sub
                   (i32.const 0)
                   (local.get $6)
                  )
                 )
                 (i32.const 1023)
                )
                (then
                 (br_if $block31
                  (i32.ge_s
                   (local.get $2)
                   (i32.const -1022)
                  )
                 )
                 (local.set $1
                  (f64.mul
                   (local.get $1)
                   (f64.const 2.004168360008973e-292)
                  )
                 )
                 (br_if $block32
                  (i32.le_u
                   (local.get $2)
                   (i32.const -1992)
                  )
                 )
                 (local.set $2
                  (i32.sub
                   (i32.const 969)
                   (local.get $6)
                  )
                 )
                 (br $block31)
                )
               )
               (local.set $1
                (f64.mul
                 (local.get $1)
                 (f64.const 8988465674311579538646525e283)
                )
               )
               (br_if $block33
                (i32.gt_u
                 (local.get $2)
                 (i32.const 2046)
                )
               )
               (local.set $2
                (i32.sub
                 (i32.const -1023)
                 (local.get $6)
                )
               )
               (br $block31)
              )
              (local.set $1
               (f64.mul
                (local.get $1)
                (f64.const 2.004168360008973e-292)
               )
              )
              (local.set $2
               (i32.add
                (select
                 (i32.const -2960)
                 (local.get $2)
                 (i32.le_u
                  (local.get $2)
                  (i32.const -2960)
                 )
                )
                (i32.const 1938)
               )
              )
              (br $block31)
             )
             (local.set $1
              (f64.mul
               (local.get $1)
               (f64.const 8988465674311579538646525e283)
              )
             )
             (local.set $2
              (i32.sub
               (select
                (i32.const 3069)
                (local.get $2)
                (i32.ge_u
                 (local.get $2)
                 (i32.const 3069)
                )
               )
               (i32.const 2046)
              )
             )
            )
            (if
             (f64.ge
              (local.tee $1
               (f64.mul
                (local.get $1)
                (f64.reinterpret_i64
                 (i64.shl
                  (i64.extend_i32_u
                   (i32.add
                    (local.get $2)
                    (i32.const 1023)
                   )
                  )
                  (i64.const 52)
                 )
                )
               )
              )
              (f64.const 16777216)
             )
             (then
              (local.set $5
               (f64.ge
                (local.tee $33
                 (f64.mul
                  (local.get $1)
                  (f64.const 5.9604644775390625e-08)
                 )
                )
                (f64.const -2147483648)
               )
              )
              (local.set $5
               (f64.ge
                (local.tee $33
                 (f64.add
                  (local.get $1)
                  (f64.mul
                   (local.tee $1
                    (f64.convert_i32_s
                     (select
                      (select
                       (i32.const 2147483647)
                       (select
                        (block $block34 (result i32)
                         (if
                          (f64.lt
                           (f64.abs
                            (local.get $33)
                           )
                           (f64.const 2147483648)
                          )
                          (then
                           (br $block34
                            (i32.trunc_f64_s
                             (local.get $33)
                            )
                           )
                          )
                         )
                         (i32.const -2147483648)
                        )
                        (i32.const -2147483648)
                        (local.get $5)
                       )
                       (f64.gt
                        (local.get $33)
                        (f64.const 2147483647)
                       )
                      )
                      (i32.const 0)
                      (f64.eq
                       (local.get $33)
                       (local.get $33)
                      )
                     )
                    )
                   )
                   (f64.const -16777216)
                  )
                 )
                )
                (f64.const -2147483648)
               )
              )
              (i32.store
               (i32.add
                (i32.add
                 (local.get $3)
                 (i32.const 480)
                )
                (i32.shl
                 (local.get $4)
                 (i32.const 2)
                )
               )
               (select
                (select
                 (i32.const 2147483647)
                 (select
                  (block $block35 (result i32)
                   (if
                    (f64.lt
                     (f64.abs
                      (local.get $33)
                     )
                     (f64.const 2147483648)
                    )
                    (then
                     (br $block35
                      (i32.trunc_f64_s
                       (local.get $33)
                      )
                     )
                    )
                   )
                   (i32.const -2147483648)
                  )
                  (i32.const -2147483648)
                  (local.get $5)
                 )
                 (f64.gt
                  (local.get $33)
                  (f64.const 2147483647)
                 )
                )
                (i32.const 0)
                (f64.eq
                 (local.get $33)
                 (local.get $33)
                )
               )
              )
              (local.set $6
               (i32.add
                (local.get $18)
                (local.get $24)
               )
              )
              (local.set $4
               (i32.add
                (local.get $4)
                (i32.const 1)
               )
              )
             )
            )
            (local.set $5
             (f64.ge
              (local.get $1)
              (f64.const -2147483648)
             )
            )
            (i32.store
             (i32.add
              (i32.add
               (local.get $3)
               (i32.const 480)
              )
              (i32.shl
               (local.get $4)
               (i32.const 2)
              )
             )
             (select
              (select
               (i32.const 2147483647)
               (select
                (block $block36 (result i32)
                 (if
                  (f64.lt
                   (f64.abs
                    (local.get $1)
                   )
                   (f64.const 2147483648)
                  )
                  (then
                   (br $block36
                    (i32.trunc_f64_s
                     (local.get $1)
                    )
                   )
                  )
                 )
                 (i32.const -2147483648)
                )
                (i32.const -2147483648)
                (local.get $5)
               )
               (f64.gt
                (local.get $1)
                (f64.const 2147483647)
               )
              )
              (i32.const 0)
              (f64.eq
               (local.get $1)
               (local.get $1)
              )
             )
            )
           )
           (local.set $1
            (f64.mul
             (block $block38 (result f64)
              (block $block39
               (block $block37
                (if
                 (i32.le_s
                  (local.get $6)
                  (i32.const 1023)
                 )
                 (then
                  (br_if $block37
                   (i32.lt_s
                    (local.get $6)
                    (i32.const -1022)
                   )
                  )
                  (br $block38
                   (f64.const 1)
                  )
                 )
                )
                (br_if $block39
                 (i32.gt_u
                  (local.get $6)
                  (i32.const 2046)
                 )
                )
                (local.set $6
                 (i32.sub
                  (local.get $6)
                  (i32.const 1023)
                 )
                )
                (br $block38
                 (f64.const 8988465674311579538646525e283)
                )
               )
               (if
                (i32.gt_u
                 (local.get $6)
                 (i32.const -1992)
                )
                (then
                 (local.set $6
                  (i32.add
                   (local.get $6)
                   (i32.const 969)
                  )
                 )
                 (br $block38
                  (f64.const 2.004168360008973e-292)
                 )
                )
               )
               (local.set $6
                (i32.add
                 (select
                  (i32.const -2960)
                  (local.get $6)
                  (i32.le_u
                   (local.get $6)
                   (i32.const -2960)
                  )
                 )
                 (i32.const 1938)
                )
               )
               (br $block38
                (f64.const 0)
               )
              )
              (local.set $6
               (i32.sub
                (select
                 (i32.const 3069)
                 (local.get $6)
                 (i32.ge_u
                  (local.get $6)
                  (i32.const 3069)
                 )
                )
                (i32.const 2046)
               )
              )
              (f64.const inf)
             )
             (f64.reinterpret_i64
              (i64.shl
               (i64.extend_i32_u
                (i32.add
                 (local.get $6)
                 (i32.const 1023)
                )
               )
               (i64.const 52)
              )
             )
            )
           )
           (local.set $9
            (if (result i32)
             (i32.and
              (local.get $4)
              (i32.const 1)
             )
             (then
              (local.get $4)
             )
             (else
              (f64.store
               (i32.add
                (i32.add
                 (local.get $3)
                 (i32.const 320)
                )
                (i32.shl
                 (local.get $4)
                 (i32.const 3)
                )
               )
               (f64.mul
                (local.get $1)
                (f64.convert_i32_s
                 (i32.load
                  (i32.add
                   (i32.add
                    (local.get $3)
                    (i32.const 480)
                   )
                   (i32.shl
                    (local.get $4)
                    (i32.const 2)
                   )
                  )
                 )
                )
               )
              )
              (local.set $1
               (f64.mul
                (local.get $1)
                (f64.const 5.9604644775390625e-08)
               )
              )
              (i32.sub
               (local.get $4)
               (i32.const 1)
              )
             )
            )
           )
           (if
            (local.get $4)
            (then
             (local.set $2
              (i32.add
               (i32.add
                (i32.shl
                 (local.get $9)
                 (i32.const 3)
                )
                (local.get $3)
               )
               (i32.const 312)
              )
             )
             (local.set $5
              (i32.add
               (i32.add
                (i32.shl
                 (local.get $9)
                 (i32.const 2)
                )
                (local.get $3)
               )
               (i32.const 476)
              )
             )
             (loop $label11
              (f64.store
               (local.get $2)
               (f64.mul
                (local.tee $33
                 (f64.mul
                  (local.get $1)
                  (f64.const 5.9604644775390625e-08)
                 )
                )
                (f64.convert_i32_s
                 (i32.load
                  (local.get $5)
                 )
                )
               )
              )
              (f64.store
               (i32.add
                (local.get $2)
                (i32.const 8)
               )
               (f64.mul
                (local.get $1)
                (f64.convert_i32_s
                 (i32.load
                  (i32.add
                   (local.get $5)
                   (i32.const 4)
                  )
                 )
                )
               )
              )
              (local.set $2
               (i32.sub
                (local.get $2)
                (i32.const 16)
               )
              )
              (local.set $5
               (i32.sub
                (local.get $5)
                (i32.const 8)
               )
              )
              (local.set $1
               (f64.mul
                (local.get $33)
                (f64.const 5.9604644775390625e-08)
               )
              )
              (br_if $label11
               (block (result i32)
                (local.set $scratch_40
                 (i32.ne
                  (local.get $9)
                  (i32.const 1)
                 )
                )
                (local.set $9
                 (i32.sub
                  (local.get $9)
                  (i32.const 2)
                 )
                )
                (local.get $scratch_40)
               )
              )
             )
            )
           )
           (local.set $12
            (i32.add
             (local.get $4)
             (i32.const 1)
            )
           )
           (local.set $8
            (i32.add
             (i32.add
              (local.get $3)
              (i32.const 320)
             )
             (i32.shl
              (local.get $4)
              (i32.const 3)
             )
            )
           )
           (local.set $2
            (local.get $4)
           )
           (loop $label13
            (block $block40
             (if
              (i32.eqz
               (local.tee $7
                (select
                 (local.get $11)
                 (local.tee $6
                  (i32.sub
                   (local.get $4)
                   (local.tee $9
                    (local.get $2)
                   )
                  )
                 )
                 (i32.gt_u
                  (local.get $6)
                  (local.get $11)
                 )
                )
               )
              )
              (then
               (local.set $5
                (i32.const 0)
               )
               (local.set $1
                (f64.const 0)
               )
               (br $block40)
              )
             )
             (local.set $14
              (i32.and
               (i32.add
                (local.get $7)
                (i32.const 1)
               )
               (i32.const -2)
              )
             )
             (local.set $1
              (f64.const 0)
             )
             (local.set $2
              (i32.const 0)
             )
             (local.set $5
              (i32.const 0)
             )
             (loop $label12
              (local.set $1
               (f64.add
                (f64.add
                 (local.get $1)
                 (f64.mul
                  (f64.load
                   (i32.add
                    (local.get $2)
                    (i32.const 1050904)
                   )
                  )
                  (f64.load
                   (local.tee $15
                    (i32.add
                     (local.get $2)
                     (local.get $8)
                    )
                   )
                  )
                 )
                )
                (f64.mul
                 (f64.load
                  (i32.add
                   (local.get $2)
                   (i32.const 1050912)
                  )
                 )
                 (f64.load
                  (i32.add
                   (local.get $15)
                   (i32.const 8)
                  )
                 )
                )
               )
              )
              (local.set $2
               (i32.add
                (local.get $2)
                (i32.const 16)
               )
              )
              (br_if $label12
               (i32.ne
                (local.get $14)
                (local.tee $5
                 (i32.add
                  (local.get $5)
                  (i32.const 2)
                 )
                )
               )
              )
             )
            )
            (f64.store
             (i32.add
              (i32.add
               (local.get $3)
               (i32.const 160)
              )
              (i32.shl
               (local.get $6)
               (i32.const 3)
              )
             )
             (if (result f64)
              (i32.and
               (local.get $7)
               (i32.const 1)
              )
              (then
               (local.get $1)
              )
              (else
               (f64.add
                (local.get $1)
                (f64.mul
                 (f64.load
                  (i32.add
                   (i32.shl
                    (local.get $5)
                    (i32.const 3)
                   )
                   (i32.const 1050904)
                  )
                 )
                 (f64.load
                  (i32.add
                   (i32.add
                    (local.get $3)
                    (i32.const 320)
                   )
                   (i32.shl
                    (i32.add
                     (local.get $5)
                     (local.get $9)
                    )
                    (i32.const 3)
                   )
                  )
                 )
                )
               )
              )
             )
            )
            (local.set $8
             (i32.sub
              (local.get $8)
              (i32.const 8)
             )
            )
            (local.set $2
             (i32.sub
              (local.get $9)
              (i32.const 1)
             )
            )
            (br_if $label13
             (local.get $9)
            )
           )
           (block $block41
            (if
             (i32.eqz
              (local.tee $9
               (i32.and
                (local.get $12)
                (i32.const 3)
               )
              )
             )
             (then
              (local.set $1
               (f64.const 0)
              )
              (local.set $5
               (local.get $4)
              )
              (br $block41)
             )
            )
            (local.set $2
             (i32.add
              (i32.add
               (local.get $3)
               (i32.const 160)
              )
              (i32.shl
               (local.get $4)
               (i32.const 3)
              )
             )
            )
            (local.set $1
             (f64.const 0)
            )
            (local.set $5
             (local.get $4)
            )
            (loop $label14
             (local.set $5
              (i32.sub
               (local.get $5)
               (i32.const 1)
              )
             )
             (local.set $1
              (f64.add
               (local.get $1)
               (f64.load
                (local.get $2)
               )
              )
             )
             (local.set $2
              (i32.sub
               (local.get $2)
               (i32.const 8)
              )
             )
             (br_if $label14
              (local.tee $9
               (i32.sub
                (local.get $9)
                (i32.const 1)
               )
              )
             )
            )
           )
           (if
            (i32.ge_u
             (local.get $4)
             (i32.const 3)
            )
            (then
             (local.set $2
              (i32.add
               (i32.add
                (i32.shl
                 (local.get $5)
                 (i32.const 3)
                )
                (local.get $3)
               )
               (i32.const 136)
              )
             )
             (loop $label15
              (local.set $1
               (f64.add
                (f64.add
                 (f64.add
                  (f64.add
                   (local.get $1)
                   (f64.load
                    (i32.add
                     (local.get $2)
                     (i32.const 24)
                    )
                   )
                  )
                  (f64.load
                   (i32.add
                    (local.get $2)
                    (i32.const 16)
                   )
                  )
                 )
                 (f64.load
                  (i32.add
                   (local.get $2)
                   (i32.const 8)
                  )
                 )
                )
                (f64.load
                 (local.get $2)
                )
               )
              )
              (local.set $2
               (i32.sub
                (local.get $2)
                (i32.const 32)
               )
              )
              (br_if $label15
               (block (result i32)
                (local.set $scratch_41
                 (i32.ne
                  (local.get $5)
                  (i32.const 3)
                 )
                )
                (local.set $5
                 (i32.sub
                  (local.get $5)
                  (i32.const 4)
                 )
                )
                (local.get $scratch_41)
               )
              )
             )
            )
           )
           (f64.store
            (local.get $17)
            (select
             (f64.neg
              (local.get $1)
             )
             (local.get $1)
             (local.get $13)
            )
           )
           (local.set $1
            (f64.sub
             (f64.load offset=160
              (local.get $3)
             )
             (local.get $1)
            )
           )
           (block $block42
            (br_if $block42
             (i32.eqz
              (local.get $4)
             )
            )
            (local.set $2
             (i32.const 1)
            )
            (loop $label16
             (local.set $1
              (f64.add
               (local.get $1)
               (f64.load
                (i32.add
                 (i32.add
                  (local.get $3)
                  (i32.const 160)
                 )
                 (i32.shl
                  (local.get $2)
                  (i32.const 3)
                 )
                )
               )
              )
             )
             (br_if $block42
              (i32.ge_u
               (local.get $2)
               (local.get $4)
              )
             )
             (br_if $label16
              (i32.le_u
               (local.tee $2
                (i32.add
                 (local.get $2)
                 (i32.lt_u
                  (local.get $2)
                  (local.get $4)
                 )
                )
               )
               (local.get $4)
              )
             )
            )
           )
           (f64.store offset=8
            (local.get $17)
            (select
             (f64.neg
              (local.get $1)
             )
             (local.get $1)
             (local.get $13)
            )
           )
           (global.set $global$0
            (i32.add
             (local.get $3)
             (i32.const 560)
            )
           )
           (local.set $4
            (i32.and
             (local.get $16)
             (i32.const 7)
            )
           )
           (if
            (i64.ge_s
             (local.get $37)
             (i64.const 0)
            )
            (then
             (i32.store offset=8
              (local.get $0)
              (local.get $4)
             )
             (f64.store offset=16
              (local.get $0)
              (f64.load offset=32
               (local.get $10)
              )
             )
             (f64.store
              (local.get $0)
              (f64.load offset=24
               (local.get $10)
              )
             )
             (br $block5)
            )
           )
           (i32.store offset=8
            (local.get $0)
            (i32.sub
             (i32.const 0)
             (local.get $4)
            )
           )
           (f64.store offset=16
            (local.get $0)
            (f64.neg
             (f64.load offset=32
              (local.get $10)
             )
            )
           )
           (f64.store
            (local.get $0)
            (f64.neg
             (f64.load offset=24
              (local.get $10)
             )
            )
           )
           (br $block5)
          )
         )
         (if
          (i32.ge_u
           (local.get $4)
           (i32.const 1075183037)
          )
          (then
           (if
            (i32.eq
             (local.get $4)
             (i32.const 1075388923)
            )
            (then
             (block $block43
              (br_if $block43
               (i64.gt_u
                (i64.and
                 (i64.reinterpret_f64
                  (local.tee $35
                   (f64.sub
                    (local.tee $1
                     (f64.add
                      (local.get $1)
                      (f64.mul
                       (local.tee $34
                        (f64.add
                         (f64.add
                          (f64.mul
                           (local.get $1)
                           (f64.const 0.6366197723675814)
                          )
                          (f64.const 6755399441055744)
                         )
                         (f64.const -6755399441055744)
                        )
                       )
                       (f64.const -1.5707963267341256)
                      )
                     )
                    )
                    (local.tee $36
                     (f64.mul
                      (local.get $34)
                      (f64.const 6.077100506506192e-11)
                     )
                    )
                   )
                  )
                 )
                 (i64.const 9218868437227405312)
                )
                (i64.const 4544132024016830463)
               )
              )
              (if
               (i64.gt_u
                (i64.and
                 (i64.reinterpret_f64
                  (local.tee $35
                   (f64.sub
                    (local.tee $33
                     (f64.sub
                      (local.get $1)
                      (local.tee $35
                       (f64.mul
                        (local.get $34)
                        (f64.const 6.077100506303966e-11)
                       )
                      )
                     )
                    )
                    (local.tee $36
                     (f64.sub
                      (f64.mul
                       (local.get $34)
                       (f64.const 2.0222662487959506e-21)
                      )
                      (f64.sub
                       (f64.sub
                        (local.get $1)
                        (local.get $33)
                       )
                       (local.get $35)
                      )
                     )
                    )
                   )
                  )
                 )
                 (i64.const 9151314442816847872)
                )
                (i64.const 4395513236313604095)
               )
               (then
                (local.set $1
                 (local.get $33)
                )
                (br $block43)
               )
              )
              (local.set $35
               (f64.sub
                (local.tee $1
                 (f64.sub
                  (local.get $33)
                  (local.tee $35
                   (f64.mul
                    (local.get $34)
                    (f64.const 2.0222662487111665e-21)
                   )
                  )
                 )
                )
                (local.tee $36
                 (f64.sub
                  (f64.mul
                   (local.get $34)
                   (f64.const 8.4784276603689e-32)
                  )
                  (f64.sub
                   (f64.sub
                    (local.get $33)
                    (local.get $1)
                   )
                   (local.get $35)
                  )
                 )
                )
               )
              )
             )
             (f64.store
              (local.get $0)
              (local.get $35)
             )
             (f64.store offset=16
              (local.get $0)
              (f64.sub
               (f64.sub
                (local.get $1)
                (local.get $35)
               )
               (local.get $36)
              )
             )
             (local.set $4
              (f64.ge
               (local.get $34)
               (f64.const -2147483648)
              )
             )
             (i32.store offset=8
              (local.get $0)
              (select
               (select
                (i32.const 2147483647)
                (select
                 (block $block44 (result i32)
                  (if
                   (f64.lt
                    (f64.abs
                     (local.get $34)
                    )
                    (f64.const 2147483648)
                   )
                   (then
                    (br $block44
                     (i32.trunc_f64_s
                      (local.get $34)
                     )
                    )
                   )
                  )
                  (i32.const -2147483648)
                 )
                 (i32.const -2147483648)
                 (local.get $4)
                )
                (f64.gt
                 (local.get $34)
                 (f64.const 2147483647)
                )
               )
               (i32.const 0)
               (f64.eq
                (local.get $34)
                (local.get $34)
               )
              )
             )
             (br $block5)
            )
           )
           (if
            (i64.ge_s
             (local.get $37)
             (i64.const 0)
            )
            (then
             (i32.store offset=8
              (local.get $0)
              (i32.const 4)
             )
             (f64.store
              (local.get $0)
              (local.tee $33
               (f64.add
                (local.tee $1
                 (f64.add
                  (local.get $1)
                  (f64.const -6.2831853069365025)
                 )
                )
                (f64.const -2.430840202602477e-10)
               )
              )
             )
             (f64.store offset=16
              (local.get $0)
              (f64.add
               (f64.sub
                (local.get $1)
                (local.get $33)
               )
               (f64.const -2.430840202602477e-10)
              )
             )
             (br $block5)
            )
           )
           (i32.store offset=8
            (local.get $0)
            (i32.const -4)
           )
           (f64.store
            (local.get $0)
            (local.tee $33
             (f64.add
              (local.tee $1
               (f64.add
                (local.get $1)
                (f64.const 6.2831853069365025)
               )
              )
              (f64.const 2.430840202602477e-10)
             )
            )
           )
           (f64.store offset=16
            (local.get $0)
            (f64.add
             (f64.sub
              (local.get $1)
              (local.get $33)
             )
             (f64.const 2.430840202602477e-10)
            )
           )
           (br $block5)
          )
         )
         (br_if $block45
          (i32.eq
           (local.get $4)
           (i32.const 1074977148)
          )
         )
         (if
          (i64.ge_s
           (local.get $37)
           (i64.const 0)
          )
          (then
           (i32.store offset=8
            (local.get $0)
            (i32.const 3)
           )
           (f64.store
            (local.get $0)
            (local.tee $33
             (f64.add
              (local.tee $1
               (f64.add
                (local.get $1)
                (f64.const -4.712388980202377)
               )
              )
              (f64.const -1.8231301519518578e-10)
             )
            )
           )
           (f64.store offset=16
            (local.get $0)
            (f64.add
             (f64.sub
              (local.get $1)
              (local.get $33)
             )
             (f64.const -1.8231301519518578e-10)
            )
           )
           (br $block5)
          )
         )
         (i32.store offset=8
          (local.get $0)
          (i32.const -3)
         )
         (f64.store
          (local.get $0)
          (local.tee $33
           (f64.add
            (local.tee $1
             (f64.add
              (local.get $1)
              (f64.const 4.712388980202377)
             )
            )
            (f64.const 1.8231301519518578e-10)
           )
          )
         )
         (f64.store offset=16
          (local.get $0)
          (f64.add
           (f64.sub
            (local.get $1)
            (local.get $33)
           )
           (f64.const 1.8231301519518578e-10)
          )
         )
         (br $block5)
        )
       )
       (br_if $block46
        (i32.eq
         (i32.and
          (local.get $5)
          (i32.const 1048575)
         )
         (i32.const 598523)
        )
       )
       (if
        (i32.ge_u
         (local.get $4)
         (i32.const 1073928573)
        )
        (then
         (if
          (i64.ge_s
           (local.get $37)
           (i64.const 0)
          )
          (then
           (i32.store offset=8
            (local.get $0)
            (i32.const 2)
           )
           (f64.store
            (local.get $0)
            (local.tee $33
             (f64.add
              (local.tee $1
               (f64.add
                (local.get $1)
                (f64.const -3.1415926534682512)
               )
              )
              (f64.const -1.2154201013012384e-10)
             )
            )
           )
           (f64.store offset=16
            (local.get $0)
            (f64.add
             (f64.sub
              (local.get $1)
              (local.get $33)
             )
             (f64.const -1.2154201013012384e-10)
            )
           )
           (br $block5)
          )
         )
         (i32.store offset=8
          (local.get $0)
          (i32.const -2)
         )
         (f64.store
          (local.get $0)
          (local.tee $33
           (f64.add
            (local.tee $1
             (f64.add
              (local.get $1)
              (f64.const 3.1415926534682512)
             )
            )
            (f64.const 1.2154201013012384e-10)
           )
          )
         )
         (f64.store offset=16
          (local.get $0)
          (f64.add
           (f64.sub
            (local.get $1)
            (local.get $33)
           )
           (f64.const 1.2154201013012384e-10)
          )
         )
         (br $block5)
        )
       )
       (br_if $block47
        (i64.ge_s
         (local.get $37)
         (i64.const 0)
        )
       )
       (i32.store offset=8
        (local.get $0)
        (i32.const -1)
       )
       (f64.store
        (local.get $0)
        (local.tee $33
         (f64.add
          (local.tee $1
           (f64.add
            (local.get $1)
            (f64.const 1.5707963267341256)
           )
          )
          (f64.const 6.077100506506192e-11)
         )
        )
       )
       (f64.store offset=16
        (local.get $0)
        (f64.add
         (f64.sub
          (local.get $1)
          (local.get $33)
         )
         (f64.const 6.077100506506192e-11)
        )
       )
       (br $block5)
      )
      (i32.store offset=8
       (local.get $0)
       (i32.const 0)
      )
      (f64.store offset=16
       (local.get $0)
       (local.tee $1
        (f64.sub
         (local.get $1)
         (local.get $1)
        )
       )
      )
      (f64.store
       (local.get $0)
       (local.get $1)
      )
      (br $block5)
     )
     (i32.store offset=8
      (local.get $0)
      (i32.const 1)
     )
     (f64.store
      (local.get $0)
      (local.tee $33
       (f64.add
        (local.tee $1
         (f64.add
          (local.get $1)
          (f64.const -1.5707963267341256)
         )
        )
        (f64.const -6.077100506506192e-11)
       )
      )
     )
     (f64.store offset=16
      (local.get $0)
      (f64.add
       (f64.sub
        (local.get $1)
        (local.get $33)
       )
       (f64.const -6.077100506506192e-11)
      )
     )
     (br $block5)
    )
    (block $block48
     (br_if $block48
      (i32.lt_s
       (i32.sub
        (local.tee $4
         (i32.shr_u
          (local.get $4)
          (i32.const 20)
         )
        )
        (i32.and
         (i32.wrap_i64
          (i64.shr_u
           (i64.reinterpret_f64
            (local.tee $35
             (f64.sub
              (local.tee $1
               (f64.add
                (local.get $1)
                (f64.mul
                 (local.tee $34
                  (f64.add
                   (f64.add
                    (f64.mul
                     (local.get $1)
                     (f64.const 0.6366197723675814)
                    )
                    (f64.const 6755399441055744)
                   )
                   (f64.const -6755399441055744)
                  )
                 )
                 (f64.const -1.5707963267341256)
                )
               )
              )
              (local.tee $36
               (f64.mul
                (local.get $34)
                (f64.const 6.077100506506192e-11)
               )
              )
             )
            )
           )
           (i64.const 52)
          )
         )
         (i32.const 2047)
        )
       )
       (i32.const 17)
      )
     )
     (if
      (i32.lt_s
       (i32.sub
        (local.get $4)
        (i32.and
         (i32.wrap_i64
          (i64.shr_u
           (i64.reinterpret_f64
            (local.tee $35
             (f64.sub
              (local.tee $33
               (f64.sub
                (local.get $1)
                (local.tee $35
                 (f64.mul
                  (local.get $34)
                  (f64.const 6.077100506303966e-11)
                 )
                )
               )
              )
              (local.tee $36
               (f64.sub
                (f64.mul
                 (local.get $34)
                 (f64.const 2.0222662487959506e-21)
                )
                (f64.sub
                 (f64.sub
                  (local.get $1)
                  (local.get $33)
                 )
                 (local.get $35)
                )
               )
              )
             )
            )
           )
           (i64.const 52)
          )
         )
         (i32.const 2047)
        )
       )
       (i32.const 50)
      )
      (then
       (local.set $1
        (local.get $33)
       )
       (br $block48)
      )
     )
     (local.set $35
      (f64.sub
       (local.tee $1
        (f64.sub
         (local.get $33)
         (local.tee $35
          (f64.mul
           (local.get $34)
           (f64.const 2.0222662487111665e-21)
          )
         )
        )
       )
       (local.tee $36
        (f64.sub
         (f64.mul
          (local.get $34)
          (f64.const 8.4784276603689e-32)
         )
         (f64.sub
          (f64.sub
           (local.get $33)
           (local.get $1)
          )
          (local.get $35)
         )
        )
       )
      )
     )
    )
    (f64.store
     (local.get $0)
     (local.get $35)
    )
    (f64.store offset=16
     (local.get $0)
     (f64.sub
      (f64.sub
       (local.get $1)
       (local.get $35)
      )
      (local.get $36)
     )
    )
    (local.set $4
     (f64.ge
      (local.get $34)
      (f64.const -2147483648)
     )
    )
    (i32.store offset=8
     (local.get $0)
     (select
      (select
       (i32.const 2147483647)
       (select
        (block $block49 (result i32)
         (if
          (f64.lt
           (f64.abs
            (local.get $34)
           )
           (f64.const 2147483648)
          )
          (then
           (br $block49
            (i32.trunc_f64_s
             (local.get $34)
            )
           )
          )
         )
         (i32.const -2147483648)
        )
        (i32.const -2147483648)
        (local.get $4)
       )
       (f64.gt
        (local.get $34)
        (f64.const 2147483647)
       )
      )
      (i32.const 0)
      (f64.eq
       (local.get $34)
       (local.get $34)
      )
     )
    )
    (br $block5)
   )
   (block $block50
    (br_if $block50
     (i64.gt_u
      (i64.and
       (i64.reinterpret_f64
        (local.tee $35
         (f64.sub
          (local.tee $1
           (f64.add
            (local.get $1)
            (f64.mul
             (local.tee $34
              (f64.add
               (f64.add
                (f64.mul
                 (local.get $1)
                 (f64.const 0.6366197723675814)
                )
                (f64.const 6755399441055744)
               )
               (f64.const -6755399441055744)
              )
             )
             (f64.const -1.5707963267341256)
            )
           )
          )
          (local.tee $36
           (f64.mul
            (local.get $34)
            (f64.const 6.077100506506192e-11)
           )
          )
         )
        )
       )
       (i64.const 9218868437227405312)
      )
      (i64.const 4544132024016830463)
     )
    )
    (if
     (i64.gt_u
      (i64.and
       (i64.reinterpret_f64
        (local.tee $35
         (f64.sub
          (local.tee $33
           (f64.sub
            (local.get $1)
            (local.tee $35
             (f64.mul
              (local.get $34)
              (f64.const 6.077100506303966e-11)
             )
            )
           )
          )
          (local.tee $36
           (f64.sub
            (f64.mul
             (local.get $34)
             (f64.const 2.0222662487959506e-21)
            )
            (f64.sub
             (f64.sub
              (local.get $1)
              (local.get $33)
             )
             (local.get $35)
            )
           )
          )
         )
        )
       )
       (i64.const 9151314442816847872)
      )
      (i64.const 4395513236313604095)
     )
     (then
      (local.set $1
       (local.get $33)
      )
      (br $block50)
     )
    )
    (local.set $35
     (f64.sub
      (local.tee $1
       (f64.sub
        (local.get $33)
        (local.tee $35
         (f64.mul
          (local.get $34)
          (f64.const 2.0222662487111665e-21)
         )
        )
       )
      )
      (local.tee $36
       (f64.sub
        (f64.mul
         (local.get $34)
         (f64.const 8.4784276603689e-32)
        )
        (f64.sub
         (f64.sub
          (local.get $33)
          (local.get $1)
         )
         (local.get $35)
        )
       )
      )
     )
    )
   )
   (f64.store
    (local.get $0)
    (local.get $35)
   )
   (f64.store offset=16
    (local.get $0)
    (f64.sub
     (f64.sub
      (local.get $1)
      (local.get $35)
     )
     (local.get $36)
    )
   )
   (local.set $4
    (f64.ge
     (local.get $34)
     (f64.const -2147483648)
    )
   )
   (i32.store offset=8
    (local.get $0)
    (select
     (select
      (i32.const 2147483647)
      (select
       (block $block51 (result i32)
        (if
         (f64.lt
          (f64.abs
           (local.get $34)
          )
          (f64.const 2147483648)
         )
         (then
          (br $block51
           (i32.trunc_f64_s
            (local.get $34)
           )
          )
         )
        )
        (i32.const -2147483648)
       )
       (i32.const -2147483648)
       (local.get $4)
      )
      (f64.gt
       (local.get $34)
       (f64.const 2147483647)
      )
     )
     (i32.const 0)
     (f64.eq
      (local.get $34)
      (local.get $34)
     )
    )
   )
  )
  (global.set $global$0
   (i32.add
    (local.get $10)
    (i32.const 48)
   )
  )
 )
 (func $2 (param $0 i32) (param $1 i32) (param $2 i32) (result i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i32)
  (local $6 i32)
  (local $7 i32)
  (local $8 i32)
  (local $9 i32)
  (local $10 i32)
  (block $block
   (if
    (i32.lt_u
     (local.get $2)
     (i32.const 16)
    )
    (then
     (local.set $3
      (local.get $0)
     )
     (br $block)
    )
   )
   (block $block1
    (br_if $block1
     (i32.le_u
      (local.tee $5
       (i32.add
        (local.get $0)
        (local.tee $6
         (i32.and
          (i32.sub
           (i32.const 0)
           (local.get $0)
          )
          (i32.const 3)
         )
        )
       )
      )
      (local.get $0)
     )
    )
    (local.set $3
     (local.get $0)
    )
    (local.set $4
     (local.get $1)
    )
    (if
     (local.get $6)
     (then
      (local.set $7
       (local.get $6)
      )
      (loop $label
       (i32.store8
        (local.get $3)
        (i32.load8_u
         (local.get $4)
        )
       )
       (local.set $4
        (i32.add
         (local.get $4)
         (i32.const 1)
        )
       )
       (local.set $3
        (i32.add
         (local.get $3)
         (i32.const 1)
        )
       )
       (br_if $label
        (local.tee $7
         (i32.sub
          (local.get $7)
          (i32.const 1)
         )
        )
       )
      )
     )
    )
    (br_if $block1
     (i32.lt_u
      (i32.sub
       (local.get $6)
       (i32.const 1)
      )
      (i32.const 7)
     )
    )
    (loop $label1
     (i32.store8
      (local.get $3)
      (i32.load8_u
       (local.get $4)
      )
     )
     (i32.store8
      (i32.add
       (local.get $3)
       (i32.const 1)
      )
      (i32.load8_u
       (i32.add
        (local.get $4)
        (i32.const 1)
       )
      )
     )
     (i32.store8
      (i32.add
       (local.get $3)
       (i32.const 2)
      )
      (i32.load8_u
       (i32.add
        (local.get $4)
        (i32.const 2)
       )
      )
     )
     (i32.store8
      (i32.add
       (local.get $3)
       (i32.const 3)
      )
      (i32.load8_u
       (i32.add
        (local.get $4)
        (i32.const 3)
       )
      )
     )
     (i32.store8
      (i32.add
       (local.get $3)
       (i32.const 4)
      )
      (i32.load8_u
       (i32.add
        (local.get $4)
        (i32.const 4)
       )
      )
     )
     (i32.store8
      (i32.add
       (local.get $3)
       (i32.const 5)
      )
      (i32.load8_u
       (i32.add
        (local.get $4)
        (i32.const 5)
       )
      )
     )
     (i32.store8
      (i32.add
       (local.get $3)
       (i32.const 6)
      )
      (i32.load8_u
       (i32.add
        (local.get $4)
        (i32.const 6)
       )
      )
     )
     (i32.store8
      (i32.add
       (local.get $3)
       (i32.const 7)
      )
      (i32.load8_u
       (i32.add
        (local.get $4)
        (i32.const 7)
       )
      )
     )
     (local.set $4
      (i32.add
       (local.get $4)
       (i32.const 8)
      )
     )
     (br_if $label1
      (i32.ne
       (local.tee $3
        (i32.add
         (local.get $3)
         (i32.const 8)
        )
       )
       (local.get $5)
      )
     )
    )
   )
   (local.set $3
    (i32.add
     (local.get $5)
     (local.tee $8
      (i32.and
       (local.tee $7
        (i32.sub
         (local.get $2)
         (local.get $6)
        )
       )
       (i32.const -4)
      )
     )
    )
   )
   (block $block2
    (if
     (i32.eqz
      (i32.and
       (local.tee $4
        (i32.add
         (local.get $1)
         (local.get $6)
        )
       )
       (i32.const 3)
      )
     )
     (then
      (br_if $block2
       (i32.le_u
        (local.get $3)
        (local.get $5)
       )
      )
      (local.set $1
       (local.get $4)
      )
      (loop $label2
       (i32.store
        (local.get $5)
        (i32.load
         (local.get $1)
        )
       )
       (local.set $1
        (i32.add
         (local.get $1)
         (i32.const 4)
        )
       )
       (br_if $label2
        (i32.lt_u
         (local.tee $5
          (i32.add
           (local.get $5)
           (i32.const 4)
          )
         )
         (local.get $3)
        )
       )
      )
      (br $block2)
     )
    )
    (br_if $block2
     (i32.le_u
      (local.get $3)
      (local.get $5)
     )
    )
    (local.set $6
     (i32.and
      (local.tee $2
       (i32.shl
        (local.get $4)
        (i32.const 3)
       )
      )
      (i32.const 24)
     )
    )
    (local.set $1
     (i32.add
      (local.tee $9
       (i32.and
        (local.get $4)
        (i32.const -4)
       )
      )
      (i32.const 4)
     )
    )
    (local.set $10
     (i32.and
      (i32.sub
       (i32.const 0)
       (local.get $2)
      )
      (i32.const 24)
     )
    )
    (local.set $2
     (i32.load
      (local.get $9)
     )
    )
    (loop $label3
     (i32.store
      (local.get $5)
      (i32.or
       (i32.shr_u
        (local.get $2)
        (local.get $6)
       )
       (i32.shl
        (local.tee $2
         (i32.load
          (local.get $1)
         )
        )
        (local.get $10)
       )
      )
     )
     (local.set $1
      (i32.add
       (local.get $1)
       (i32.const 4)
      )
     )
     (br_if $label3
      (i32.lt_u
       (local.tee $5
        (i32.add
         (local.get $5)
         (i32.const 4)
        )
       )
       (local.get $3)
      )
     )
    )
   )
   (local.set $2
    (i32.and
     (local.get $7)
     (i32.const 3)
    )
   )
   (local.set $1
    (i32.add
     (local.get $4)
     (local.get $8)
    )
   )
  )
  (block $block3
   (br_if $block3
    (i32.ge_u
     (local.get $3)
     (local.tee $6
      (i32.add
       (local.get $2)
       (local.get $3)
      )
     )
    )
   )
   (if
    (local.tee $4
     (i32.and
      (local.get $2)
      (i32.const 7)
     )
    )
    (then
     (loop $label4
      (i32.store8
       (local.get $3)
       (i32.load8_u
        (local.get $1)
       )
      )
      (local.set $1
       (i32.add
        (local.get $1)
        (i32.const 1)
       )
      )
      (local.set $3
       (i32.add
        (local.get $3)
        (i32.const 1)
       )
      )
      (br_if $label4
       (local.tee $4
        (i32.sub
         (local.get $4)
         (i32.const 1)
        )
       )
      )
     )
    )
   )
   (br_if $block3
    (i32.lt_u
     (i32.sub
      (local.get $2)
      (i32.const 1)
     )
     (i32.const 7)
    )
   )
   (loop $label5
    (i32.store8
     (local.get $3)
     (i32.load8_u
      (local.get $1)
     )
    )
    (i32.store8
     (i32.add
      (local.get $3)
      (i32.const 1)
     )
     (i32.load8_u
      (i32.add
       (local.get $1)
       (i32.const 1)
      )
     )
    )
    (i32.store8
     (i32.add
      (local.get $3)
      (i32.const 2)
     )
     (i32.load8_u
      (i32.add
       (local.get $1)
       (i32.const 2)
      )
     )
    )
    (i32.store8
     (i32.add
      (local.get $3)
      (i32.const 3)
     )
     (i32.load8_u
      (i32.add
       (local.get $1)
       (i32.const 3)
      )
     )
    )
    (i32.store8
     (i32.add
      (local.get $3)
      (i32.const 4)
     )
     (i32.load8_u
      (i32.add
       (local.get $1)
       (i32.const 4)
      )
     )
    )
    (i32.store8
     (i32.add
      (local.get $3)
      (i32.const 5)
     )
     (i32.load8_u
      (i32.add
       (local.get $1)
       (i32.const 5)
      )
     )
    )
    (i32.store8
     (i32.add
      (local.get $3)
      (i32.const 6)
     )
     (i32.load8_u
      (i32.add
       (local.get $1)
       (i32.const 6)
      )
     )
    )
    (i32.store8
     (i32.add
      (local.get $3)
      (i32.const 7)
     )
     (i32.load8_u
      (i32.add
       (local.get $1)
       (i32.const 7)
      )
     )
    )
    (local.set $1
     (i32.add
      (local.get $1)
      (i32.const 8)
     )
    )
    (br_if $label5
     (i32.ne
      (local.tee $3
       (i32.add
        (local.get $3)
        (i32.const 8)
       )
      )
      (local.get $6)
     )
    )
   )
  )
  (local.get $0)
 )
 (func $3 (param $0 i32)
  (local $1 i32)
  (local $2 i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i32)
  (local.set $2
   (i32.add
    (local.tee $1
     (i32.sub
      (local.get $0)
      (i32.const 8)
     )
    )
    (local.tee $0
     (i32.and
      (local.tee $3
       (i32.load
        (i32.sub
         (local.get $0)
         (i32.const 4)
        )
       )
      )
      (i32.const -8)
     )
    )
   )
  )
  (block $block1
   (block $block
    (br_if $block
     (i32.and
      (local.get $3)
      (i32.const 1)
     )
    )
    (br_if $block1
     (i32.eqz
      (i32.and
       (local.get $3)
       (i32.const 2)
      )
     )
    )
    (local.set $0
     (i32.add
      (local.tee $3
       (i32.load
        (local.get $1)
       )
      )
      (local.get $0)
     )
    )
    (if
     (i32.eq
      (local.tee $1
       (i32.sub
        (local.get $1)
        (local.get $3)
       )
      )
      (i32.load
       (i32.const 1051452)
      )
     )
     (then
      (br_if $block
       (i32.ne
        (i32.and
         (i32.load offset=4
          (local.get $2)
         )
         (i32.const 3)
        )
        (i32.const 3)
       )
      )
      (i32.store
       (i32.const 1051444)
       (local.get $0)
      )
      (i32.store offset=4
       (local.get $2)
       (i32.and
        (i32.load offset=4
         (local.get $2)
        )
        (i32.const -2)
       )
      )
      (i32.store offset=4
       (local.get $1)
       (i32.or
        (local.get $0)
        (i32.const 1)
       )
      )
      (i32.store
       (local.get $2)
       (local.get $0)
      )
      (return)
     )
    )
    (call $8
     (local.get $1)
     (local.get $3)
    )
   )
   (block $block6
    (block $block5
     (block $block3
      (block $block2
       (block $block4
        (if
         (i32.eqz
          (i32.and
           (local.tee $3
            (i32.load offset=4
             (local.get $2)
            )
           )
           (i32.const 2)
          )
         )
         (then
          (br_if $block2
           (i32.eq
            (local.get $2)
            (i32.load
             (i32.const 1051456)
            )
           )
          )
          (br_if $block3
           (i32.eq
            (local.get $2)
            (i32.load
             (i32.const 1051452)
            )
           )
          )
          (call $8
           (local.get $2)
           (local.tee $2
            (i32.and
             (local.get $3)
             (i32.const -8)
            )
           )
          )
          (i32.store offset=4
           (local.get $1)
           (i32.or
            (local.tee $0
             (i32.add
              (local.get $0)
              (local.get $2)
             )
            )
            (i32.const 1)
           )
          )
          (i32.store
           (i32.add
            (local.get $0)
            (local.get $1)
           )
           (local.get $0)
          )
          (br_if $block4
           (i32.ne
            (local.get $1)
            (i32.load
             (i32.const 1051452)
            )
           )
          )
          (i32.store
           (i32.const 1051444)
           (local.get $0)
          )
          (return)
         )
        )
        (i32.store offset=4
         (local.get $2)
         (i32.and
          (local.get $3)
          (i32.const -2)
         )
        )
        (i32.store offset=4
         (local.get $1)
         (i32.or
          (local.get $0)
          (i32.const 1)
         )
        )
        (i32.store
         (i32.add
          (local.get $0)
          (local.get $1)
         )
         (local.get $0)
        )
       )
       (br_if $block5
        (i32.lt_u
         (local.get $0)
         (i32.const 256)
        )
       )
       (call $10
        (local.get $1)
        (local.get $0)
       )
       (local.set $1
        (i32.const 0)
       )
       (i32.store
        (i32.const 1051476)
        (local.tee $0
         (i32.sub
          (i32.load
           (i32.const 1051476)
          )
          (i32.const 1)
         )
        )
       )
       (br_if $block1
        (local.get $0)
       )
       (if
        (local.tee $0
         (i32.load
          (i32.const 1051164)
         )
        )
        (then
         (loop $label
          (local.set $1
           (i32.add
            (local.get $1)
            (i32.const 1)
           )
          )
          (br_if $label
           (local.tee $0
            (i32.load offset=8
             (local.get $0)
            )
           )
          )
         )
        )
       )
       (i32.store
        (i32.const 1051476)
        (select
         (i32.const 4095)
         (local.get $1)
         (i32.le_u
          (local.get $1)
          (i32.const 4095)
         )
        )
       )
       (return)
      )
      (i32.store
       (i32.const 1051456)
       (local.get $1)
      )
      (i32.store
       (i32.const 1051448)
       (local.tee $0
        (i32.add
         (i32.load
          (i32.const 1051448)
         )
         (local.get $0)
        )
       )
      )
      (i32.store offset=4
       (local.get $1)
       (i32.or
        (local.get $0)
        (i32.const 1)
       )
      )
      (if
       (i32.eq
        (i32.load
         (i32.const 1051452)
        )
        (local.get $1)
       )
       (then
        (i32.store
         (i32.const 1051444)
         (i32.const 0)
        )
        (i32.store
         (i32.const 1051452)
         (i32.const 0)
        )
       )
      )
      (br_if $block1
       (i32.le_u
        (local.get $0)
        (local.tee $3
         (i32.load
          (i32.const 1051468)
         )
        )
       )
      )
      (br_if $block1
       (i32.eqz
        (local.tee $2
         (i32.load
          (i32.const 1051456)
         )
        )
       )
      )
      (local.set $0
       (i32.const 0)
      )
      (br_if $block6
       (i32.lt_u
        (local.tee $4
         (i32.load
          (i32.const 1051448)
         )
        )
        (i32.const 41)
       )
      )
      (local.set $1
       (i32.const 1051156)
      )
      (loop $label1
       (if
        (i32.ge_u
         (local.get $2)
         (local.tee $5
          (i32.load
           (local.get $1)
          )
         )
        )
        (then
         (br_if $block6
          (i32.lt_u
           (local.get $2)
           (i32.add
            (local.get $5)
            (i32.load offset=4
             (local.get $1)
            )
           )
          )
         )
        )
       )
       (local.set $1
        (i32.load offset=8
         (local.get $1)
        )
       )
       (br $label1)
      )
      (unreachable)
     )
     (i32.store
      (i32.const 1051452)
      (local.get $1)
     )
     (i32.store
      (i32.const 1051444)
      (local.tee $0
       (i32.add
        (i32.load
         (i32.const 1051444)
        )
        (local.get $0)
       )
      )
     )
     (i32.store offset=4
      (local.get $1)
      (i32.or
       (local.get $0)
       (i32.const 1)
      )
     )
     (i32.store
      (i32.add
       (local.get $0)
       (local.get $1)
      )
      (local.get $0)
     )
     (return)
    )
    (local.set $2
     (i32.add
      (i32.and
       (local.get $0)
       (i32.const 248)
      )
      (i32.const 1051172)
     )
    )
    (local.set $0
     (block $block7 (result i32)
      (if
       (i32.eqz
        (i32.and
         (local.tee $3
          (i32.load
           (i32.const 1051436)
          )
         )
         (local.tee $0
          (i32.shl
           (i32.const 1)
           (i32.shr_u
            (local.get $0)
            (i32.const 3)
           )
          )
         )
        )
       )
       (then
        (i32.store
         (i32.const 1051436)
         (i32.or
          (local.get $0)
          (local.get $3)
         )
        )
        (br $block7
         (local.get $2)
        )
       )
      )
      (i32.load offset=8
       (local.get $2)
      )
     )
    )
    (i32.store offset=8
     (local.get $2)
     (local.get $1)
    )
    (i32.store offset=12
     (local.get $0)
     (local.get $1)
    )
    (i32.store offset=12
     (local.get $1)
     (local.get $2)
    )
    (i32.store offset=8
     (local.get $1)
     (local.get $0)
    )
    (return)
   )
   (if
    (local.tee $1
     (i32.load
      (i32.const 1051164)
     )
    )
    (then
     (loop $label2
      (local.set $0
       (i32.add
        (local.get $0)
        (i32.const 1)
       )
      )
      (br_if $label2
       (local.tee $1
        (i32.load offset=8
         (local.get $1)
        )
       )
      )
     )
    )
   )
   (i32.store
    (i32.const 1051476)
    (select
     (i32.const 4095)
     (local.get $0)
     (i32.le_u
      (local.get $0)
      (i32.const 4095)
     )
    )
   )
   (br_if $block1
    (i32.ge_u
     (local.get $3)
     (local.get $4)
    )
   )
   (i32.store
    (i32.const 1051468)
    (i32.const -1)
   )
  )
 )
 (func $4 (param $0 i32) (param $1 i32) (param $2 i32) (result i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i32)
  (local $6 i32)
  (local $7 i32)
  (local $8 i32)
  (local $9 i32)
  (local $10 i32)
  (local $11 i32)
  (local $12 i32)
  (local $scratch i32)
  (global.set $global$0
   (local.tee $3
    (i32.sub
     (global.get $global$0)
     (i32.const 48)
    )
   )
  )
  (i32.store offset=44
   (local.get $3)
   (local.get $1)
  )
  (i32.store offset=40
   (local.get $3)
   (local.get $0)
  )
  (i32.store8 offset=36
   (local.get $3)
   (i32.const 3)
  )
  (i64.store offset=28 align=4
   (local.get $3)
   (i64.const 32)
  )
  (i32.store offset=20
   (local.get $3)
   (i32.const 0)
  )
  (i32.store offset=12
   (local.get $3)
   (i32.const 0)
  )
  (local.set $scratch
   (block $block9 (result i32)
    (block $block8
     (block $block1
      (block $block
       (if
        (i32.eqz
         (local.tee $10
          (i32.load offset=16
           (local.get $2)
          )
         )
        )
        (then
         (br_if $block
          (i32.eqz
           (local.tee $0
            (i32.load offset=12
             (local.get $2)
            )
           )
          )
         )
         (local.set $4
          (i32.add
           (local.tee $1
            (i32.load offset=8
             (local.get $2)
            )
           )
           (i32.shl
            (local.get $0)
            (i32.const 3)
           )
          )
         )
         (local.set $7
          (i32.add
           (i32.and
            (i32.sub
             (local.get $0)
             (i32.const 1)
            )
            (i32.const 536870911)
           )
           (i32.const 1)
          )
         )
         (local.set $0
          (i32.load
           (local.get $2)
          )
         )
         (loop $label
          (if
           (local.tee $5
            (i32.load
             (i32.add
              (local.get $0)
              (i32.const 4)
             )
            )
           )
           (then
            (br_if $block1
             (call_indirect $0 (type $2)
              (i32.load offset=40
               (local.get $3)
              )
              (i32.load
               (local.get $0)
              )
              (local.get $5)
              (i32.load offset=12
               (i32.load offset=44
                (local.get $3)
               )
              )
             )
            )
           )
          )
          (br_if $block1
           (call_indirect $0 (type $1)
            (i32.load
             (local.get $1)
            )
            (i32.add
             (local.get $3)
             (i32.const 12)
            )
            (i32.load
             (i32.add
              (local.get $1)
              (i32.const 4)
             )
            )
           )
          )
          (local.set $0
           (i32.add
            (local.get $0)
            (i32.const 8)
           )
          )
          (br_if $label
           (i32.ne
            (local.tee $1
             (i32.add
              (local.get $1)
              (i32.const 8)
             )
            )
            (local.get $4)
           )
          )
         )
         (br $block)
        )
       )
       (br_if $block
        (i32.eqz
         (local.tee $0
          (i32.load offset=20
           (local.get $2)
          )
         )
        )
       )
       (local.set $11
        (i32.shl
         (local.get $0)
         (i32.const 5)
        )
       )
       (local.set $7
        (i32.add
         (i32.and
          (i32.sub
           (local.get $0)
           (i32.const 1)
          )
          (i32.const 134217727)
         )
         (i32.const 1)
        )
       )
       (local.set $5
        (i32.load offset=8
         (local.get $2)
        )
       )
       (local.set $0
        (i32.load
         (local.get $2)
        )
       )
       (loop $label1
        (if
         (local.tee $1
          (i32.load
           (i32.add
            (local.get $0)
            (i32.const 4)
           )
          )
         )
         (then
          (br_if $block1
           (call_indirect $0 (type $2)
            (i32.load offset=40
             (local.get $3)
            )
            (i32.load
             (local.get $0)
            )
            (local.get $1)
            (i32.load offset=12
             (i32.load offset=44
              (local.get $3)
             )
            )
           )
          )
         )
        )
        (i32.store offset=28
         (local.get $3)
         (i32.load
          (i32.add
           (local.tee $1
            (i32.add
             (local.get $8)
             (local.get $10)
            )
           )
           (i32.const 16)
          )
         )
        )
        (i32.store8 offset=36
         (local.get $3)
         (i32.load8_u
          (i32.add
           (local.get $1)
           (i32.const 28)
          )
         )
        )
        (i32.store offset=32
         (local.get $3)
         (i32.load
          (i32.add
           (local.get $1)
           (i32.const 24)
          )
         )
        )
        (local.set $4
         (i32.load
          (i32.add
           (local.get $1)
           (i32.const 12)
          )
         )
        )
        (local.set $9
         (i32.const 0)
        )
        (local.set $6
         (i32.const 0)
        )
        (block $block3
         (block $block4
          (block $block2
           (br_table $block2 $block3 $block4
            (i32.sub
             (i32.load
              (i32.add
               (local.get $1)
               (i32.const 8)
              )
             )
             (i32.const 1)
            )
           )
          )
          (br_if $block3
           (i32.load
            (local.tee $12
             (i32.add
              (i32.shl
               (local.get $4)
               (i32.const 3)
              )
              (local.get $5)
             )
            )
           )
          )
          (local.set $4
           (i32.load offset=4
            (local.get $12)
           )
          )
         )
         (local.set $6
          (i32.const 1)
         )
        )
        (i32.store offset=16
         (local.get $3)
         (local.get $4)
        )
        (i32.store offset=12
         (local.get $3)
         (local.get $6)
        )
        (local.set $4
         (i32.load
          (i32.add
           (local.get $1)
           (i32.const 4)
          )
         )
        )
        (block $block6
         (block $block7
          (block $block5
           (br_table $block5 $block6 $block7
            (i32.sub
             (i32.load
              (local.get $1)
             )
             (i32.const 1)
            )
           )
          )
          (br_if $block6
           (i32.load
            (local.tee $6
             (i32.add
              (i32.shl
               (local.get $4)
               (i32.const 3)
              )
              (local.get $5)
             )
            )
           )
          )
          (local.set $4
           (i32.load offset=4
            (local.get $6)
           )
          )
         )
         (local.set $9
          (i32.const 1)
         )
        )
        (i32.store offset=24
         (local.get $3)
         (local.get $4)
        )
        (i32.store offset=20
         (local.get $3)
         (local.get $9)
        )
        (br_if $block1
         (call_indirect $0 (type $1)
          (i32.load
           (local.tee $1
            (i32.add
             (local.get $5)
             (i32.shl
              (i32.load
               (i32.add
                (local.get $1)
                (i32.const 20)
               )
              )
              (i32.const 3)
             )
            )
           )
          )
          (i32.add
           (local.get $3)
           (i32.const 12)
          )
          (i32.load
           (i32.add
            (local.get $1)
            (i32.const 4)
           )
          )
         )
        )
        (local.set $0
         (i32.add
          (local.get $0)
          (i32.const 8)
         )
        )
        (br_if $label1
         (i32.ne
          (local.get $11)
          (local.tee $8
           (i32.add
            (local.get $8)
            (i32.const 32)
           )
          )
         )
        )
       )
      )
      (br_if $block8
       (i32.ge_u
        (local.get $7)
        (i32.load offset=4
         (local.get $2)
        )
       )
      )
      (br_if $block8
       (i32.eqz
        (call_indirect $0 (type $2)
         (i32.load offset=40
          (local.get $3)
         )
         (i32.load
          (local.tee $0
           (i32.add
            (i32.load
             (local.get $2)
            )
            (i32.shl
             (local.get $7)
             (i32.const 3)
            )
           )
          )
         )
         (i32.load offset=4
          (local.get $0)
         )
         (i32.load offset=12
          (i32.load offset=44
           (local.get $3)
          )
         )
        )
       )
      )
     )
     (br $block9
      (i32.const 1)
     )
    )
    (i32.const 0)
   )
  )
  (global.set $global$0
   (i32.add
    (local.get $3)
    (i32.const 48)
   )
  )
  (local.get $scratch)
 )
 (func $5 (param $0 i32) (param $1 i32)
  (local $2 i32)
  (local $3 i32)
  (local.set $2
   (i32.add
    (local.get $0)
    (local.get $1)
   )
  )
  (block $block1
   (block $block
    (br_if $block
     (i32.and
      (local.tee $3
       (i32.load offset=4
        (local.get $0)
       )
      )
      (i32.const 1)
     )
    )
    (br_if $block1
     (i32.eqz
      (i32.and
       (local.get $3)
       (i32.const 2)
      )
     )
    )
    (local.set $1
     (i32.add
      (local.tee $3
       (i32.load
        (local.get $0)
       )
      )
      (local.get $1)
     )
    )
    (if
     (i32.eq
      (local.tee $0
       (i32.sub
        (local.get $0)
        (local.get $3)
       )
      )
      (i32.load
       (i32.const 1051452)
      )
     )
     (then
      (br_if $block
       (i32.ne
        (i32.and
         (i32.load offset=4
          (local.get $2)
         )
         (i32.const 3)
        )
        (i32.const 3)
       )
      )
      (i32.store
       (i32.const 1051444)
       (local.get $1)
      )
      (i32.store offset=4
       (local.get $2)
       (i32.and
        (i32.load offset=4
         (local.get $2)
        )
        (i32.const -2)
       )
      )
      (i32.store offset=4
       (local.get $0)
       (i32.or
        (local.get $1)
        (i32.const 1)
       )
      )
      (i32.store
       (local.get $2)
       (local.get $1)
      )
      (br $block1)
     )
    )
    (call $8
     (local.get $0)
     (local.get $3)
    )
   )
   (block $block3
    (block $block2
     (block $block4
      (if
       (i32.eqz
        (i32.and
         (local.tee $3
          (i32.load offset=4
           (local.get $2)
          )
         )
         (i32.const 2)
        )
       )
       (then
        (br_if $block2
         (i32.eq
          (local.get $2)
          (i32.load
           (i32.const 1051456)
          )
         )
        )
        (br_if $block3
         (i32.eq
          (local.get $2)
          (i32.load
           (i32.const 1051452)
          )
         )
        )
        (call $8
         (local.get $2)
         (local.tee $2
          (i32.and
           (local.get $3)
           (i32.const -8)
          )
         )
        )
        (i32.store offset=4
         (local.get $0)
         (i32.or
          (local.tee $1
           (i32.add
            (local.get $1)
            (local.get $2)
           )
          )
          (i32.const 1)
         )
        )
        (i32.store
         (i32.add
          (local.get $0)
          (local.get $1)
         )
         (local.get $1)
        )
        (br_if $block4
         (i32.ne
          (local.get $0)
          (i32.load
           (i32.const 1051452)
          )
         )
        )
        (i32.store
         (i32.const 1051444)
         (local.get $1)
        )
        (return)
       )
      )
      (i32.store offset=4
       (local.get $2)
       (i32.and
        (local.get $3)
        (i32.const -2)
       )
      )
      (i32.store offset=4
       (local.get $0)
       (i32.or
        (local.get $1)
        (i32.const 1)
       )
      )
      (i32.store
       (i32.add
        (local.get $0)
        (local.get $1)
       )
       (local.get $1)
      )
     )
     (if
      (i32.ge_u
       (local.get $1)
       (i32.const 256)
      )
      (then
       (call $10
        (local.get $0)
        (local.get $1)
       )
       (return)
      )
     )
     (local.set $2
      (i32.add
       (i32.and
        (local.get $1)
        (i32.const 248)
       )
       (i32.const 1051172)
      )
     )
     (local.set $1
      (block $block5 (result i32)
       (if
        (i32.eqz
         (i32.and
          (local.tee $3
           (i32.load
            (i32.const 1051436)
           )
          )
          (local.tee $1
           (i32.shl
            (i32.const 1)
            (i32.shr_u
             (local.get $1)
             (i32.const 3)
            )
           )
          )
         )
        )
        (then
         (i32.store
          (i32.const 1051436)
          (i32.or
           (local.get $1)
           (local.get $3)
          )
         )
         (br $block5
          (local.get $2)
         )
        )
       )
       (i32.load offset=8
        (local.get $2)
       )
      )
     )
     (i32.store offset=8
      (local.get $2)
      (local.get $0)
     )
     (i32.store offset=12
      (local.get $1)
      (local.get $0)
     )
     (i32.store offset=12
      (local.get $0)
      (local.get $2)
     )
     (i32.store offset=8
      (local.get $0)
      (local.get $1)
     )
     (return)
    )
    (i32.store
     (i32.const 1051456)
     (local.get $0)
    )
    (i32.store
     (i32.const 1051448)
     (local.tee $1
      (i32.add
       (i32.load
        (i32.const 1051448)
       )
       (local.get $1)
      )
     )
    )
    (i32.store offset=4
     (local.get $0)
     (i32.or
      (local.get $1)
      (i32.const 1)
     )
    )
    (br_if $block1
     (i32.ne
      (local.get $0)
      (i32.load
       (i32.const 1051452)
      )
     )
    )
    (i32.store
     (i32.const 1051444)
     (i32.const 0)
    )
    (i32.store
     (i32.const 1051452)
     (i32.const 0)
    )
    (return)
   )
   (i32.store
    (i32.const 1051452)
    (local.get $0)
   )
   (i32.store
    (i32.const 1051444)
    (local.tee $1
     (i32.add
      (i32.load
       (i32.const 1051444)
      )
      (local.get $1)
     )
    )
   )
   (i32.store offset=4
    (local.get $0)
    (i32.or
     (local.get $1)
     (i32.const 1)
    )
   )
   (i32.store
    (i32.add
     (local.get $0)
     (local.get $1)
    )
    (local.get $1)
   )
  )
 )
 (func $6 (param $0 i32) (param $1 i32)
  (local $2 i32)
  (local $3 i32)
  (local $4 i32)
  (if
   (i32.ge_u
    (local.get $1)
    (i32.const 16)
   )
   (then
    (block $block
     (br_if $block
      (i32.le_u
       (local.tee $2
        (i32.add
         (local.get $0)
         (local.tee $3
          (i32.and
           (i32.sub
            (i32.const 0)
            (local.get $0)
           )
           (i32.const 3)
          )
         )
        )
       )
       (local.get $0)
      )
     )
     (if
      (local.get $3)
      (then
       (local.set $4
        (local.get $3)
       )
       (loop $label
        (i32.store8
         (local.get $0)
         (i32.const 0)
        )
        (local.set $0
         (i32.add
          (local.get $0)
          (i32.const 1)
         )
        )
        (br_if $label
         (local.tee $4
          (i32.sub
           (local.get $4)
           (i32.const 1)
          )
         )
        )
       )
      )
     )
     (br_if $block
      (i32.lt_u
       (i32.sub
        (local.get $3)
        (i32.const 1)
       )
       (i32.const 7)
      )
     )
     (loop $label1
      (i32.store8
       (local.get $0)
       (i32.const 0)
      )
      (i32.store8
       (i32.add
        (local.get $0)
        (i32.const 7)
       )
       (i32.const 0)
      )
      (i32.store8
       (i32.add
        (local.get $0)
        (i32.const 6)
       )
       (i32.const 0)
      )
      (i32.store8
       (i32.add
        (local.get $0)
        (i32.const 5)
       )
       (i32.const 0)
      )
      (i32.store8
       (i32.add
        (local.get $0)
        (i32.const 4)
       )
       (i32.const 0)
      )
      (i32.store8
       (i32.add
        (local.get $0)
        (i32.const 3)
       )
       (i32.const 0)
      )
      (i32.store8
       (i32.add
        (local.get $0)
        (i32.const 2)
       )
       (i32.const 0)
      )
      (i32.store8
       (i32.add
        (local.get $0)
        (i32.const 1)
       )
       (i32.const 0)
      )
      (br_if $label1
       (i32.ne
        (local.tee $0
         (i32.add
          (local.get $0)
          (i32.const 8)
         )
        )
        (local.get $2)
       )
      )
     )
    )
    (if
     (i32.gt_u
      (local.tee $0
       (i32.add
        (local.get $2)
        (i32.and
         (local.tee $1
          (i32.sub
           (local.get $1)
           (local.get $3)
          )
         )
         (i32.const -4)
        )
       )
      )
      (local.get $2)
     )
     (then
      (loop $label2
       (i32.store
        (local.get $2)
        (i32.const 0)
       )
       (br_if $label2
        (i32.lt_u
         (local.tee $2
          (i32.add
           (local.get $2)
           (i32.const 4)
          )
         )
         (local.get $0)
        )
       )
      )
     )
    )
    (local.set $1
     (i32.and
      (local.get $1)
      (i32.const 3)
     )
    )
   )
  )
  (block $block1
   (br_if $block1
    (i32.ge_u
     (local.get $0)
     (local.tee $3
      (i32.add
       (local.get $0)
       (local.get $1)
      )
     )
    )
   )
   (if
    (local.tee $2
     (i32.and
      (local.get $1)
      (i32.const 7)
     )
    )
    (then
     (loop $label3
      (i32.store8
       (local.get $0)
       (i32.const 0)
      )
      (local.set $0
       (i32.add
        (local.get $0)
        (i32.const 1)
       )
      )
      (br_if $label3
       (local.tee $2
        (i32.sub
         (local.get $2)
         (i32.const 1)
        )
       )
      )
     )
    )
   )
   (br_if $block1
    (i32.lt_u
     (i32.sub
      (local.get $1)
      (i32.const 1)
     )
     (i32.const 7)
    )
   )
   (loop $label4
    (i32.store8
     (local.get $0)
     (i32.const 0)
    )
    (i32.store8
     (i32.add
      (local.get $0)
      (i32.const 7)
     )
     (i32.const 0)
    )
    (i32.store8
     (i32.add
      (local.get $0)
      (i32.const 6)
     )
     (i32.const 0)
    )
    (i32.store8
     (i32.add
      (local.get $0)
      (i32.const 5)
     )
     (i32.const 0)
    )
    (i32.store8
     (i32.add
      (local.get $0)
      (i32.const 4)
     )
     (i32.const 0)
    )
    (i32.store8
     (i32.add
      (local.get $0)
      (i32.const 3)
     )
     (i32.const 0)
    )
    (i32.store8
     (i32.add
      (local.get $0)
      (i32.const 2)
     )
     (i32.const 0)
    )
    (i32.store8
     (i32.add
      (local.get $0)
      (i32.const 1)
     )
     (i32.const 0)
    )
    (br_if $label4
     (i32.ne
      (local.tee $0
       (i32.add
        (local.get $0)
        (i32.const 8)
       )
      )
      (local.get $3)
     )
    )
   )
  )
 )
 (func $7 (param $0 i32) (param $1 i32) (result i32)
  (local $2 i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i32)
  (local $6 i32)
  (block $block
   (br_if $block
    (i32.le_u
     (i32.sub
      (i32.const -65587)
      (local.tee $0
       (select
        (i32.const 16)
        (local.get $0)
        (i32.le_u
         (local.get $0)
         (i32.const 16)
        )
       )
      )
     )
     (local.get $1)
    )
   )
   (br_if $block
    (i32.eqz
     (local.tee $2
      (call $0
       (i32.add
        (i32.add
         (local.get $0)
         (local.tee $4
          (select
           (i32.const 16)
           (i32.and
            (i32.add
             (local.get $1)
             (i32.const 11)
            )
            (i32.const -8)
           )
           (i32.lt_u
            (local.get $1)
            (i32.const 11)
           )
          )
         )
        )
        (i32.const 12)
       )
      )
     )
    )
   )
   (local.set $1
    (i32.sub
     (local.get $2)
     (i32.const 8)
    )
   )
   (block $block1
    (if
     (i32.eqz
      (i32.and
       (local.tee $3
        (i32.sub
         (local.get $0)
         (i32.const 1)
        )
       )
       (local.get $2)
      )
     )
     (then
      (local.set $0
       (local.get $1)
      )
      (br $block1)
     )
    )
    (local.set $3
     (i32.sub
      (i32.and
       (local.tee $6
        (i32.load
         (local.tee $5
          (i32.sub
           (local.get $2)
           (i32.const 4)
          )
         )
        )
       )
       (i32.const -8)
      )
      (local.tee $2
       (i32.sub
        (local.tee $0
         (i32.add
          (local.tee $2
           (i32.sub
            (i32.and
             (i32.add
              (local.get $2)
              (local.get $3)
             )
             (i32.sub
              (i32.const 0)
              (local.get $0)
             )
            )
            (i32.const 8)
           )
          )
          (select
           (local.get $0)
           (i32.const 0)
           (i32.le_u
            (i32.sub
             (local.get $2)
             (local.get $1)
            )
            (i32.const 16)
           )
          )
         )
        )
        (local.get $1)
       )
      )
     )
    )
    (if
     (i32.and
      (local.get $6)
      (i32.const 3)
     )
     (then
      (i32.store offset=4
       (local.get $0)
       (i32.or
        (i32.or
         (local.get $3)
         (i32.and
          (i32.load offset=4
           (local.get $0)
          )
          (i32.const 1)
         )
        )
        (i32.const 2)
       )
      )
      (i32.store offset=4
       (local.tee $3
        (i32.add
         (local.get $0)
         (local.get $3)
        )
       )
       (i32.or
        (i32.load offset=4
         (local.get $3)
        )
        (i32.const 1)
       )
      )
      (i32.store
       (local.get $5)
       (i32.or
        (i32.or
         (local.get $2)
         (i32.and
          (i32.load
           (local.get $5)
          )
          (i32.const 1)
         )
        )
        (i32.const 2)
       )
      )
      (i32.store offset=4
       (local.tee $3
        (i32.add
         (local.get $1)
         (local.get $2)
        )
       )
       (i32.or
        (i32.load offset=4
         (local.get $3)
        )
        (i32.const 1)
       )
      )
      (call $5
       (local.get $1)
       (local.get $2)
      )
      (br $block1)
     )
    )
    (local.set $1
     (i32.load
      (local.get $1)
     )
    )
    (i32.store offset=4
     (local.get $0)
     (local.get $3)
    )
    (i32.store
     (local.get $0)
     (i32.add
      (local.get $1)
      (local.get $2)
     )
    )
   )
   (block $block2
    (br_if $block2
     (i32.eqz
      (i32.and
       (local.tee $1
        (i32.load offset=4
         (local.get $0)
        )
       )
       (i32.const 3)
      )
     )
    )
    (br_if $block2
     (i32.le_u
      (local.tee $2
       (i32.and
        (local.get $1)
        (i32.const -8)
       )
      )
      (i32.add
       (local.get $4)
       (i32.const 16)
      )
     )
    )
    (i32.store offset=4
     (local.get $0)
     (i32.or
      (i32.or
       (local.get $4)
       (i32.and
        (local.get $1)
        (i32.const 1)
       )
      )
      (i32.const 2)
     )
    )
    (i32.store offset=4
     (local.tee $1
      (i32.add
       (local.get $0)
       (local.get $4)
      )
     )
     (i32.or
      (local.tee $4
       (i32.sub
        (local.get $2)
        (local.get $4)
       )
      )
      (i32.const 3)
     )
    )
    (i32.store offset=4
     (local.tee $2
      (i32.add
       (local.get $0)
       (local.get $2)
      )
     )
     (i32.or
      (i32.load offset=4
       (local.get $2)
      )
      (i32.const 1)
     )
    )
    (call $5
     (local.get $1)
     (local.get $4)
    )
   )
   (local.set $3
    (i32.add
     (local.get $0)
     (i32.const 8)
    )
   )
  )
  (local.get $3)
 )
 (func $8 (param $0 i32) (param $1 i32)
  (local $2 i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i32)
  (local.set $2
   (i32.load offset=12
    (local.get $0)
   )
  )
  (block $block2
   (block $block3
    (if
     (i32.ge_u
      (local.get $1)
      (i32.const 256)
     )
     (then
      (local.set $3
       (i32.load offset=24
        (local.get $0)
       )
      )
      (block $block1
       (block $block
        (if
         (i32.eq
          (local.get $0)
          (local.get $2)
         )
         (then
          (br_if $block
           (local.tee $1
            (i32.load
             (i32.add
              (local.get $0)
              (select
               (i32.const 20)
               (i32.const 16)
               (local.tee $2
                (i32.load offset=20
                 (local.get $0)
                )
               )
              )
             )
            )
           )
          )
          (local.set $2
           (i32.const 0)
          )
          (br $block1)
         )
        )
        (i32.store offset=12
         (local.tee $1
          (i32.load offset=8
           (local.get $0)
          )
         )
         (local.get $2)
        )
        (i32.store offset=8
         (local.get $2)
         (local.get $1)
        )
        (br $block1)
       )
       (local.set $4
        (select
         (i32.add
          (local.get $0)
          (i32.const 20)
         )
         (i32.add
          (local.get $0)
          (i32.const 16)
         )
         (local.get $2)
        )
       )
       (loop $label
        (local.set $5
         (local.get $4)
        )
        (local.set $4
         (select
          (i32.add
           (local.tee $2
            (local.get $1)
           )
           (i32.const 20)
          )
          (i32.add
           (local.get $2)
           (i32.const 16)
          )
          (local.tee $1
           (i32.load offset=20
            (local.get $2)
           )
          )
         )
        )
        (br_if $label
         (local.tee $1
          (i32.load
           (i32.add
            (local.get $2)
            (select
             (i32.const 20)
             (i32.const 16)
             (local.get $1)
            )
           )
          )
         )
        )
       )
       (i32.store
        (local.get $5)
        (i32.const 0)
       )
      )
      (br_if $block2
       (i32.eqz
        (local.get $3)
       )
      )
      (if
       (i32.ne
        (local.get $0)
        (i32.load
         (local.tee $1
          (i32.add
           (i32.shl
            (i32.load offset=28
             (local.get $0)
            )
            (i32.const 2)
           )
           (i32.const 1051028)
          )
         )
        )
       )
       (then
        (i32.store
         (i32.add
          (local.get $3)
          (select
           (i32.const 16)
           (i32.const 20)
           (i32.eq
            (i32.load offset=16
             (local.get $3)
            )
            (local.get $0)
           )
          )
         )
         (local.get $2)
        )
        (br_if $block2
         (i32.eqz
          (local.get $2)
         )
        )
        (br $block3)
       )
      )
      (i32.store
       (local.get $1)
       (local.get $2)
      )
      (br_if $block3
       (local.get $2)
      )
      (i32.store
       (i32.const 1051440)
       (i32.and
        (i32.load
         (i32.const 1051440)
        )
        (i32.rotl
         (i32.const -2)
         (i32.load offset=28
          (local.get $0)
         )
        )
       )
      )
      (br $block2)
     )
    )
    (if
     (i32.ne
      (local.tee $0
       (i32.load offset=8
        (local.get $0)
       )
      )
      (local.get $2)
     )
     (then
      (i32.store offset=12
       (local.get $0)
       (local.get $2)
      )
      (i32.store offset=8
       (local.get $2)
       (local.get $0)
      )
      (return)
     )
    )
    (i32.store
     (i32.const 1051436)
     (i32.and
      (i32.load
       (i32.const 1051436)
      )
      (i32.rotl
       (i32.const -2)
       (i32.shr_u
        (local.get $1)
        (i32.const 3)
       )
      )
     )
    )
    (return)
   )
   (i32.store offset=24
    (local.get $2)
    (local.get $3)
   )
   (if
    (local.tee $1
     (i32.load offset=16
      (local.get $0)
     )
    )
    (then
     (i32.store offset=16
      (local.get $2)
      (local.get $1)
     )
     (i32.store offset=24
      (local.get $1)
      (local.get $2)
     )
    )
   )
   (br_if $block2
    (i32.eqz
     (local.tee $0
      (i32.load offset=20
       (local.get $0)
      )
     )
    )
   )
   (i32.store offset=20
    (local.get $2)
    (local.get $0)
   )
   (i32.store offset=24
    (local.get $0)
    (local.get $2)
   )
  )
 )
 (func $9 (param $0 i32) (param $1 i32) (result i32)
  (local $2 i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i32)
  (local $6 i32)
  (local $7 i32)
  (global.set $global$0
   (local.tee $3
    (i32.sub
     (global.get $global$0)
     (i32.const 16)
    )
   )
  )
  (block $block1
   (if
    (i32.ge_u
     (local.get $1)
     (i32.const 128)
    )
    (then
     (i32.store offset=12
      (local.get $3)
      (i32.const 0)
     )
     (local.set $1
      (block $block (result i32)
       (if
        (i32.ge_u
         (local.get $1)
         (i32.const 2048)
        )
        (then
         (if
          (i32.ge_u
           (local.get $1)
           (i32.const 65536)
          )
          (then
           (i32.store8 offset=15
            (local.get $3)
            (i32.or
             (i32.and
              (local.get $1)
              (i32.const 63)
             )
             (i32.const 128)
            )
           )
           (i32.store8 offset=12
            (local.get $3)
            (i32.or
             (i32.shr_u
              (local.get $1)
              (i32.const 18)
             )
             (i32.const 240)
            )
           )
           (i32.store8 offset=14
            (local.get $3)
            (i32.or
             (i32.and
              (i32.shr_u
               (local.get $1)
               (i32.const 6)
              )
              (i32.const 63)
             )
             (i32.const 128)
            )
           )
           (i32.store8 offset=13
            (local.get $3)
            (i32.or
             (i32.and
              (i32.shr_u
               (local.get $1)
               (i32.const 12)
              )
              (i32.const 63)
             )
             (i32.const 128)
            )
           )
           (br $block
            (i32.const 4)
           )
          )
         )
         (i32.store8 offset=14
          (local.get $3)
          (i32.or
           (i32.and
            (local.get $1)
            (i32.const 63)
           )
           (i32.const 128)
          )
         )
         (i32.store8 offset=12
          (local.get $3)
          (i32.or
           (i32.shr_u
            (local.get $1)
            (i32.const 12)
           )
           (i32.const 224)
          )
         )
         (i32.store8 offset=13
          (local.get $3)
          (i32.or
           (i32.and
            (i32.shr_u
             (local.get $1)
             (i32.const 6)
            )
            (i32.const 63)
           )
           (i32.const 128)
          )
         )
         (br $block
          (i32.const 3)
         )
        )
       )
       (i32.store8 offset=13
        (local.get $3)
        (i32.or
         (i32.and
          (local.get $1)
          (i32.const 63)
         )
         (i32.const 128)
        )
       )
       (i32.store8 offset=12
        (local.get $3)
        (i32.or
         (i32.shr_u
          (local.get $1)
          (i32.const 6)
         )
         (i32.const 192)
        )
       )
       (i32.const 2)
      )
     )
     (if
      (i32.gt_u
       (local.get $1)
       (i32.sub
        (i32.load
         (local.get $0)
        )
        (local.tee $2
         (i32.load offset=8
          (local.get $0)
         )
        )
       )
      )
      (then
       (call $12
        (local.get $0)
        (local.get $2)
        (local.get $1)
       )
       (local.set $2
        (i32.load offset=8
         (local.get $0)
        )
       )
      )
     )
     (drop
      (call $2
       (i32.add
        (i32.load offset=4
         (local.get $0)
        )
        (local.get $2)
       )
       (i32.add
        (local.get $3)
        (i32.const 12)
       )
       (local.get $1)
      )
     )
     (i32.store offset=8
      (local.get $0)
      (i32.add
       (local.get $1)
       (local.get $2)
      )
     )
     (br $block1)
    )
   )
   (if
    (i32.eq
     (local.tee $6
      (i32.load offset=8
       (local.get $0)
      )
     )
     (i32.load
      (local.get $0)
     )
    )
    (then
     (global.set $global$0
      (local.tee $2
       (i32.sub
        (global.get $global$0)
        (i32.const 32)
       )
      )
     )
     (if
      (i32.lt_s
       (local.tee $4
        (select
         (i32.const 8)
         (local.tee $4
          (select
           (local.tee $4
            (i32.add
             (local.tee $5
              (i32.load
               (local.get $0)
              )
             )
             (i32.const 1)
            )
           )
           (local.tee $7
            (i32.shl
             (local.get $5)
             (i32.const 1)
            )
           )
           (i32.gt_u
            (local.get $4)
            (local.get $7)
           )
          )
         )
         (i32.le_u
          (local.get $4)
          (i32.const 8)
         )
        )
       )
       (i32.const 0)
      )
      (then
       (call $29
        (i32.const 0)
        (i32.const 0)
        (i32.const 1049724)
       )
       (unreachable)
      )
     )
     (i32.store offset=24
      (local.get $2)
      (if (result i32)
       (local.get $5)
       (then
        (i32.store offset=28
         (local.get $2)
         (local.get $5)
        )
        (i32.store offset=20
         (local.get $2)
         (i32.load offset=4
          (local.get $0)
         )
        )
        (i32.const 1)
       )
       (else
        (i32.const 0)
       )
      )
     )
     (call $16
      (i32.add
       (local.get $2)
       (i32.const 8)
      )
      (i32.const 1)
      (local.get $4)
      (i32.add
       (local.get $2)
       (i32.const 20)
      )
     )
     (if
      (i32.eq
       (i32.load offset=8
        (local.get $2)
       )
       (i32.const 1)
      )
      (then
       (call $29
        (i32.load offset=12
         (local.get $2)
        )
        (i32.load offset=16
         (local.get $2)
        )
        (i32.const 1049724)
       )
       (unreachable)
      )
     )
     (local.set $5
      (i32.load offset=12
       (local.get $2)
      )
     )
     (i32.store
      (local.get $0)
      (local.get $4)
     )
     (i32.store offset=4
      (local.get $0)
      (local.get $5)
     )
     (global.set $global$0
      (i32.add
       (local.get $2)
       (i32.const 32)
      )
     )
    )
   )
   (i32.store8
    (i32.add
     (i32.load offset=4
      (local.get $0)
     )
     (local.get $6)
    )
    (local.get $1)
   )
   (i32.store offset=8
    (local.get $0)
    (i32.add
     (local.get $6)
     (i32.const 1)
    )
   )
  )
  (global.set $global$0
   (i32.add
    (local.get $3)
    (i32.const 16)
   )
  )
  (i32.const 0)
 )
 (func $10 (param $0 i32) (param $1 i32)
  (local $2 i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i32)
  (i64.store offset=16 align=4
   (local.get $0)
   (i64.const 0)
  )
  (i32.store offset=28
   (local.get $0)
   (local.tee $2
    (block $block (result i32)
     (drop
      (br_if $block
       (i32.const 0)
       (i32.lt_u
        (local.get $1)
        (i32.const 256)
       )
      )
     )
     (drop
      (br_if $block
       (i32.const 31)
       (i32.gt_u
        (local.get $1)
        (i32.const 16777215)
       )
      )
     )
     (i32.add
      (i32.sub
       (i32.and
        (i32.shr_u
         (local.get $1)
         (i32.sub
          (i32.const 6)
          (local.tee $3
           (i32.clz
            (i32.shr_u
             (local.get $1)
             (i32.const 8)
            )
           )
          )
         )
        )
        (i32.const 1)
       )
       (i32.shl
        (local.get $3)
        (i32.const 1)
       )
      )
      (i32.const 62)
     )
    )
   )
  )
  (local.set $4
   (i32.add
    (i32.shl
     (local.get $2)
     (i32.const 2)
    )
    (i32.const 1051028)
   )
  )
  (if
   (i32.eqz
    (i32.and
     (local.tee $3
      (i32.shl
       (i32.const 1)
       (local.get $2)
      )
     )
     (i32.load
      (i32.const 1051440)
     )
    )
   )
   (then
    (i32.store
     (local.get $4)
     (local.get $0)
    )
    (i32.store offset=24
     (local.get $0)
     (local.get $4)
    )
    (i32.store offset=12
     (local.get $0)
     (local.get $0)
    )
    (i32.store offset=8
     (local.get $0)
     (local.get $0)
    )
    (i32.store
     (i32.const 1051440)
     (i32.or
      (i32.load
       (i32.const 1051440)
      )
      (local.get $3)
     )
    )
    (return)
   )
  )
  (block $block2
   (block $block1
    (if
     (i32.eq
      (local.get $1)
      (i32.and
       (i32.load offset=4
        (local.tee $3
         (i32.load
          (local.get $4)
         )
        )
       )
       (i32.const -8)
      )
     )
     (then
      (local.set $2
       (local.get $3)
      )
      (br $block1)
     )
    )
    (local.set $5
     (i32.shl
      (local.get $1)
      (select
       (i32.sub
        (i32.const 25)
        (i32.shr_u
         (local.get $2)
         (i32.const 1)
        )
       )
       (i32.const 0)
       (i32.ne
        (local.get $2)
        (i32.const 31)
       )
      )
     )
    )
    (loop $label
     (br_if $block2
      (i32.eqz
       (local.tee $2
        (i32.load
         (local.tee $4
          (i32.add
           (i32.add
            (local.get $3)
            (i32.and
             (i32.shr_u
              (local.get $5)
              (i32.const 29)
             )
             (i32.const 4)
            )
           )
           (i32.const 16)
          )
         )
        )
       )
      )
     )
     (local.set $5
      (i32.shl
       (local.get $5)
       (i32.const 1)
      )
     )
     (local.set $3
      (local.get $2)
     )
     (br_if $label
      (i32.ne
       (i32.and
        (i32.load offset=4
         (local.get $2)
        )
        (i32.const -8)
       )
       (local.get $1)
      )
     )
    )
   )
   (i32.store offset=12
    (local.tee $1
     (i32.load offset=8
      (local.get $2)
     )
    )
    (local.get $0)
   )
   (i32.store offset=8
    (local.get $2)
    (local.get $0)
   )
   (i32.store offset=24
    (local.get $0)
    (i32.const 0)
   )
   (i32.store offset=12
    (local.get $0)
    (local.get $2)
   )
   (i32.store offset=8
    (local.get $0)
    (local.get $1)
   )
   (return)
  )
  (i32.store
   (local.get $4)
   (local.get $0)
  )
  (i32.store offset=24
   (local.get $0)
   (local.get $3)
  )
  (i32.store offset=12
   (local.get $0)
   (local.get $0)
  )
  (i32.store offset=8
   (local.get $0)
   (local.get $0)
  )
 )
 (func $11 (param $0 i32) (param $1 i32)
  (local $2 i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i64)
  (global.set $global$0
   (local.tee $2
    (i32.add
     (global.get $global$0)
     (i32.const -64)
    )
   )
  )
  (if
   (i32.eq
    (i32.load
     (local.get $1)
    )
    (i32.const -2147483648)
   )
   (then
    (local.set $3
     (i32.load offset=12
      (local.get $1)
     )
    )
    (i32.store
     (local.tee $4
      (i32.add
       (local.get $2)
       (i32.const 36)
      )
     )
     (i32.const 0)
    )
    (i64.store offset=28 align=4
     (local.get $2)
     (i64.const 4294967296)
    )
    (i64.store
     (i32.add
      (local.get $2)
      (i32.const 48)
     )
     (i64.load align=4
      (i32.add
       (local.tee $3
        (i32.load
         (local.get $3)
        )
       )
       (i32.const 8)
      )
     )
    )
    (i64.store
     (i32.add
      (local.get $2)
      (i32.const 56)
     )
     (i64.load align=4
      (i32.add
       (local.get $3)
       (i32.const 16)
      )
     )
    )
    (i64.store offset=40
     (local.get $2)
     (i64.load align=4
      (local.get $3)
     )
    )
    (drop
     (call $4
      (i32.add
       (local.get $2)
       (i32.const 28)
      )
      (i32.const 1049832)
      (i32.add
       (local.get $2)
       (i32.const 40)
      )
     )
    )
    (i32.store
     (i32.add
      (local.get $2)
      (i32.const 24)
     )
     (local.tee $3
      (i32.load
       (local.get $4)
      )
     )
    )
    (i64.store offset=16
     (local.get $2)
     (local.tee $5
      (i64.load offset=28 align=4
       (local.get $2)
      )
     )
    )
    (i32.store
     (i32.add
      (local.get $1)
      (i32.const 8)
     )
     (local.get $3)
    )
    (i64.store align=4
     (local.get $1)
     (local.get $5)
    )
   )
  )
  (local.set $5
   (i64.load align=4
    (local.get $1)
   )
  )
  (i64.store align=4
   (local.get $1)
   (i64.const 4294967296)
  )
  (i32.store
   (local.tee $3
    (i32.add
     (local.get $2)
     (i32.const 8)
    )
   )
   (i32.load
    (local.tee $1
     (i32.add
      (local.get $1)
      (i32.const 8)
     )
    )
   )
  )
  (i32.store
   (local.get $1)
   (i32.const 0)
  )
  (drop
   (i32.load8_u
    (i32.const 1050997)
   )
  )
  (i64.store
   (local.get $2)
   (local.get $5)
  )
  (if
   (i32.eqz
    (local.tee $1
     (call $37
      (i32.const 12)
      (i32.const 4)
     )
    )
   )
   (then
    (call $46
     (i32.const 4)
     (i32.const 12)
    )
    (unreachable)
   )
  )
  (i64.store align=4
   (local.get $1)
   (i64.load
    (local.get $2)
   )
  )
  (i32.store
   (i32.add
    (local.get $1)
    (i32.const 8)
   )
   (i32.load
    (local.get $3)
   )
  )
  (i32.store offset=4
   (local.get $0)
   (i32.const 1050116)
  )
  (i32.store
   (local.get $0)
   (local.get $1)
  )
  (global.set $global$0
   (i32.sub
    (local.get $2)
    (i32.const -64)
   )
  )
 )
 (func $12 (param $0 i32) (param $1 i32) (param $2 i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i32)
  (local $6 i32)
  (local $7 i64)
  (global.set $global$0
   (local.tee $3
    (i32.sub
     (global.get $global$0)
     (i32.const 32)
    )
   )
  )
  (block $block1
   (block $block
    (if
     (i32.gt_u
      (local.get $1)
      (local.tee $2
       (i32.add
        (local.get $1)
        (local.get $2)
       )
      )
     )
     (then
      (local.set $1
       (i32.const 0)
      )
      (br $block)
     )
    )
    (local.set $1
     (i32.const 0)
    )
    (br_if $block
     (i32.eqz
      (i64.eqz
       (i64.shr_u
        (local.tee $7
         (i64.extend_i32_u
          (local.tee $4
           (select
            (i32.const 8)
            (local.tee $2
             (select
              (local.get $2)
              (local.tee $4
               (i32.shl
                (local.tee $5
                 (i32.load
                  (local.get $0)
                 )
                )
                (i32.const 1)
               )
              )
              (i32.gt_u
               (local.get $2)
               (local.get $4)
              )
             )
            )
            (i32.le_u
             (local.get $2)
             (i32.const 8)
            )
           )
          )
         )
        )
        (i64.const 32)
       )
      )
     )
    )
    (br_if $block
     (i32.gt_u
      (local.tee $6
       (i32.wrap_i64
        (local.get $7)
       )
      )
      (i32.const 2147483647)
     )
    )
    (i32.store offset=24
     (local.get $3)
     (if (result i32)
      (local.get $5)
      (then
       (i32.store offset=28
        (local.get $3)
        (local.get $5)
       )
       (i32.store offset=20
        (local.get $3)
        (i32.load offset=4
         (local.get $0)
        )
       )
       (i32.const 1)
      )
      (else
       (i32.const 0)
      )
     )
    )
    (call $16
     (i32.add
      (local.get $3)
      (i32.const 8)
     )
     (i32.const 1)
     (local.get $6)
     (i32.add
      (local.get $3)
      (i32.const 20)
     )
    )
    (br_if $block1
     (i32.ne
      (i32.load offset=8
       (local.get $3)
      )
      (i32.const 1)
     )
    )
    (local.set $2
     (i32.load offset=16
      (local.get $3)
     )
    )
    (local.set $1
     (i32.load offset=12
      (local.get $3)
     )
    )
   )
   (call $29
    (local.get $1)
    (local.get $2)
    (i32.const 1049816)
   )
   (unreachable)
  )
  (local.set $1
   (i32.load offset=12
    (local.get $3)
   )
  )
  (i32.store
   (local.get $0)
   (local.get $4)
  )
  (i32.store offset=4
   (local.get $0)
   (local.get $1)
  )
  (global.set $global$0
   (i32.add
    (local.get $3)
    (i32.const 32)
   )
  )
 )
 (func $13 (param $0 i32) (param $1 i32)
  (local $2 i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i32)
  (local $6 i32)
  (global.set $global$0
   (local.tee $2
    (i32.sub
     (global.get $global$0)
     (i32.const 32)
    )
   )
  )
  (if
   (i32.gt_u
    (local.tee $3
     (select
      (local.tee $3
       (i32.add
        (local.tee $5
         (i32.load
          (local.get $0)
         )
        )
        (i32.const 1)
       )
      )
      (local.tee $6
       (i32.shl
        (local.get $5)
        (i32.const 1)
       )
      )
      (i32.gt_u
       (local.get $3)
       (local.get $6)
      )
     )
    )
    (i32.const 268435455)
   )
   (then
    (call $29
     (i32.const 0)
     (i32.const 0)
     (local.get $1)
    )
    (unreachable)
   )
  )
  (block $block
   (call $29
    (if (result i32)
     (i32.le_u
      (local.tee $6
       (i32.shl
        (local.tee $3
         (select
          (i32.const 4)
          (local.get $3)
          (i32.le_u
           (local.get $3)
           (i32.const 4)
          )
         )
        )
        (i32.const 4)
       )
      )
      (i32.const 2147483640)
     )
     (then
      (i32.store offset=24
       (local.get $2)
       (if (result i32)
        (local.get $5)
        (then
         (i32.store offset=28
          (local.get $2)
          (i32.shl
           (local.get $5)
           (i32.const 4)
          )
         )
         (i32.store offset=20
          (local.get $2)
          (i32.load offset=4
           (local.get $0)
          )
         )
         (i32.const 8)
        )
        (else
         (local.get $4)
        )
       )
      )
      (call $16
       (i32.add
        (local.get $2)
        (i32.const 8)
       )
       (i32.const 8)
       (local.get $6)
       (i32.add
        (local.get $2)
        (i32.const 20)
       )
      )
      (br_if $block
       (i32.ne
        (i32.load offset=8
         (local.get $2)
        )
        (i32.const 1)
       )
      )
      (local.set $4
       (i32.load offset=16
        (local.get $2)
       )
      )
      (i32.load offset=12
       (local.get $2)
      )
     )
     (else
      (local.get $4)
     )
    )
    (local.get $4)
    (local.get $1)
   )
   (unreachable)
  )
  (local.set $1
   (i32.load offset=12
    (local.get $2)
   )
  )
  (i32.store
   (local.get $0)
   (local.get $3)
  )
  (i32.store offset=4
   (local.get $0)
   (local.get $1)
  )
  (global.set $global$0
   (i32.add
    (local.get $2)
    (i32.const 32)
   )
  )
 )
 (func $14 (param $0 i32) (param $1 i32)
  (local $2 i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i64)
  (global.set $global$0
   (local.tee $2
    (i32.sub
     (global.get $global$0)
     (i32.const 48)
    )
   )
  )
  (if
   (i32.eq
    (i32.load
     (local.get $1)
    )
    (i32.const -2147483648)
   )
   (then
    (local.set $3
     (i32.load offset=12
      (local.get $1)
     )
    )
    (i32.store
     (local.tee $4
      (i32.add
       (local.get $2)
       (i32.const 20)
      )
     )
     (i32.const 0)
    )
    (i64.store offset=12 align=4
     (local.get $2)
     (i64.const 4294967296)
    )
    (i64.store
     (i32.add
      (local.get $2)
      (i32.const 32)
     )
     (i64.load align=4
      (i32.add
       (local.tee $3
        (i32.load
         (local.get $3)
        )
       )
       (i32.const 8)
      )
     )
    )
    (i64.store
     (i32.add
      (local.get $2)
      (i32.const 40)
     )
     (i64.load align=4
      (i32.add
       (local.get $3)
       (i32.const 16)
      )
     )
    )
    (i64.store offset=24
     (local.get $2)
     (i64.load align=4
      (local.get $3)
     )
    )
    (drop
     (call $4
      (i32.add
       (local.get $2)
       (i32.const 12)
      )
      (i32.const 1049832)
      (i32.add
       (local.get $2)
       (i32.const 24)
      )
     )
    )
    (i32.store
     (i32.add
      (local.get $2)
      (i32.const 8)
     )
     (local.tee $3
      (i32.load
       (local.get $4)
      )
     )
    )
    (i64.store
     (local.get $2)
     (local.tee $5
      (i64.load offset=12 align=4
       (local.get $2)
      )
     )
    )
    (i32.store
     (i32.add
      (local.get $1)
      (i32.const 8)
     )
     (local.get $3)
    )
    (i64.store align=4
     (local.get $1)
     (local.get $5)
    )
   )
  )
  (i32.store offset=4
   (local.get $0)
   (i32.const 1050116)
  )
  (i32.store
   (local.get $0)
   (local.get $1)
  )
  (global.set $global$0
   (i32.add
    (local.get $2)
    (i32.const 48)
   )
  )
 )
 (func $15 (param $0 i32) (param $1 i32) (param $2 i32) (param $3 i32) (param $4 i32)
  (local $5 i32)
  (local $6 i32)
  (global.set $global$0
   (local.tee $5
    (i32.sub
     (global.get $global$0)
     (i32.const 32)
    )
   )
  )
  (i32.store
   (i32.const 1051024)
   (i32.add
    (local.tee $6
     (i32.load
      (i32.const 1051024)
     )
    )
    (i32.const 1)
   )
  )
  (block $block1
   (if
    (i32.ne
     (local.tee $6
      (i32.and
       (block $block (result i32)
        (drop
         (br_if $block
          (i32.const 0)
          (i32.lt_s
           (local.get $6)
           (i32.const 0)
          )
         )
        )
        (drop
         (br_if $block
          (i32.const 1)
          (i32.load8_u
           (i32.const 1051484)
          )
         )
        )
        (i32.store8
         (i32.const 1051484)
         (i32.const 1)
        )
        (i32.store
         (i32.const 1051480)
         (i32.add
          (i32.load
           (i32.const 1051480)
          )
          (i32.const 1)
         )
        )
        (i32.const 2)
       )
       (i32.const 255)
      )
     )
     (i32.const 2)
    )
    (then
     (br_if $block1
      (i32.eqz
       (i32.and
        (local.get $6)
        (i32.const 1)
       )
      )
     )
     (call_indirect $0 (type $0)
      (i32.add
       (local.get $5)
       (i32.const 8)
      )
      (local.get $0)
      (i32.load offset=24
       (local.get $1)
      )
     )
     (unreachable)
    )
   )
   (br_if $block1
    (i32.lt_s
     (local.tee $6
      (i32.load
       (i32.const 1051012)
      )
     )
     (i32.const 0)
    )
   )
   (i32.store
    (i32.const 1051012)
    (i32.add
     (local.get $6)
     (i32.const 1)
    )
   )
   (i32.store
    (i32.const 1051012)
    (if (result i32)
     (i32.load
      (i32.const 1051016)
     )
     (then
      (call_indirect $0 (type $0)
       (local.get $5)
       (local.get $0)
       (i32.load offset=20
        (local.get $1)
       )
      )
      (i32.store8 offset=29
       (local.get $5)
       (local.get $4)
      )
      (i32.store8 offset=28
       (local.get $5)
       (local.get $3)
      )
      (i32.store offset=24
       (local.get $5)
       (local.get $2)
      )
      (i64.store offset=16 align=4
       (local.get $5)
       (i64.load
        (local.get $5)
       )
      )
      (call_indirect $0 (type $0)
       (i32.load
        (i32.const 1051016)
       )
       (i32.add
        (local.get $5)
        (i32.const 16)
       )
       (i32.load offset=20
        (i32.load
         (i32.const 1051020)
        )
       )
      )
      (i32.sub
       (i32.load
        (i32.const 1051012)
       )
       (i32.const 1)
      )
     )
     (else
      (local.get $6)
     )
    )
   )
   (i32.store8
    (i32.const 1051484)
    (i32.const 0)
   )
   (br_if $block1
    (i32.eqz
     (local.get $3)
    )
   )
   (unreachable)
  )
  (unreachable)
 )
 (func $16 (param $0 i32) (param $1 i32) (param $2 i32) (param $3 i32)
  (local $4 i32)
  (block $block2
   (if
    (i32.ge_s
     (local.get $2)
     (i32.const 0)
    )
    (then
     (if
      (local.tee $3
       (block $block1 (result i32)
        (if
         (i32.load offset=4
          (local.get $3)
         )
         (then
          (block $block
           (if
            (i32.eqz
             (local.tee $4
              (i32.load offset=8
               (local.get $3)
              )
             )
            )
            (then
             (br $block)
            )
           )
           (br $block1
            (call $31
             (i32.load
              (local.get $3)
             )
             (local.get $4)
             (local.get $1)
             (local.get $2)
            )
           )
          )
         )
        )
        (drop
         (br_if $block1
          (local.get $1)
          (i32.eqz
           (local.get $2)
          )
         )
        )
        (drop
         (i32.load8_u
          (i32.const 1050997)
         )
        )
        (call $37
         (local.get $2)
         (local.get $1)
        )
       )
      )
      (then
       (i32.store offset=8
        (local.get $0)
        (local.get $2)
       )
       (i32.store offset=4
        (local.get $0)
        (local.get $3)
       )
       (i32.store
        (local.get $0)
        (i32.const 0)
       )
       (return)
      )
     )
     (i32.store offset=8
      (local.get $0)
      (local.get $2)
     )
     (i32.store offset=4
      (local.get $0)
      (local.get $1)
     )
     (br $block2)
    )
   )
   (i32.store offset=4
    (local.get $0)
    (i32.const 0)
   )
  )
  (i32.store
   (local.get $0)
   (i32.const 1)
  )
 )
 (func $17 (param $0 i32) (param $1 i32) (result i32)
  (local $2 i32)
  (local $scratch i32)
  (global.set $global$0
   (local.tee $2
    (i32.sub
     (global.get $global$0)
     (i32.const 32)
    )
   )
  )
  (local.set $scratch
   (block $block (result i32)
    (if
     (i32.ne
      (i32.load
       (local.get $0)
      )
      (i32.const -2147483648)
     )
     (then
      (br $block
       (call $33
        (local.get $1)
        (i32.load offset=4
         (local.get $0)
        )
        (i32.load offset=8
         (local.get $0)
        )
       )
      )
     )
    )
    (i64.store
     (i32.add
      (local.get $2)
      (i32.const 16)
     )
     (i64.load align=4
      (i32.add
       (local.tee $0
        (i32.load
         (i32.load offset=12
          (local.get $0)
         )
        )
       )
       (i32.const 8)
      )
     )
    )
    (i64.store
     (i32.add
      (local.get $2)
      (i32.const 24)
     )
     (i64.load align=4
      (i32.add
       (local.get $0)
       (i32.const 16)
      )
     )
    )
    (i64.store offset=8
     (local.get $2)
     (i64.load align=4
      (local.get $0)
     )
    )
    (call $4
     (i32.load offset=28
      (local.get $1)
     )
     (i32.load offset=32
      (local.get $1)
     )
     (i32.add
      (local.get $2)
      (i32.const 8)
     )
    )
   )
  )
  (global.set $global$0
   (i32.add
    (local.get $2)
    (i32.const 32)
   )
  )
  (local.get $scratch)
 )
 (func $18 (param $0 i32) (param $1 i32) (param $2 i32)
  (local $3 i32)
  (local $4 i64)
  (global.set $global$0
   (local.tee $3
    (i32.sub
     (global.get $global$0)
     (i32.const 48)
    )
   )
  )
  (i32.store offset=4
   (local.get $3)
   (local.get $1)
  )
  (i32.store
   (local.get $3)
   (local.get $0)
  )
  (i32.store offset=12
   (local.get $3)
   (i32.const 2)
  )
  (i32.store offset=8
   (local.get $3)
   (i32.const 1050284)
  )
  (i64.store offset=20 align=4
   (local.get $3)
   (i64.const 2)
  )
  (i64.store offset=40
   (local.get $3)
   (i64.or
    (local.tee $4
     (i64.const 4294967296)
    )
    (i64.extend_i32_u
     (local.get $3)
    )
   )
  )
  (i64.store offset=32
   (local.get $3)
   (i64.or
    (local.get $4)
    (i64.extend_i32_u
     (i32.add
      (local.get $3)
      (i32.const 4)
     )
    )
   )
  )
  (i32.store offset=16
   (local.get $3)
   (i32.add
    (local.get $3)
    (i32.const 32)
   )
  )
  (call $26
   (i32.add
    (local.get $3)
    (i32.const 8)
   )
   (local.get $2)
  )
  (unreachable)
 )
 (func $19 (param $0 i32) (param $1 i32)
  (global.set $global$0
   (local.tee $0
    (i32.sub
     (global.get $global$0)
     (i32.const 48)
    )
   )
  )
  (if
   (i32.load8_u
    (i32.const 1050996)
   )
   (then
    (i32.store offset=12
     (local.get $0)
     (i32.const 2)
    )
    (i32.store offset=8
     (local.get $0)
     (i32.const 1050060)
    )
    (i64.store offset=20 align=4
     (local.get $0)
     (i64.const 1)
    )
    (i32.store offset=44
     (local.get $0)
     (local.get $1)
    )
    (i64.store offset=32
     (local.get $0)
     (i64.or
      (i64.extend_i32_u
       (i32.add
        (local.get $0)
        (i32.const 44)
       )
      )
      (i64.const 4294967296)
     )
    )
    (i32.store offset=16
     (local.get $0)
     (i32.add
      (local.get $0)
      (i32.const 32)
     )
    )
    (call $26
     (i32.add
      (local.get $0)
      (i32.const 8)
     )
     (i32.const 1050100)
    )
    (unreachable)
   )
  )
  (global.set $global$0
   (i32.add
    (local.get $0)
    (i32.const 48)
   )
  )
 )
 (func $20 (param $0 i32) (param $1 i32) (param $2 i32) (result i32)
  (local $3 i32)
  (if
   (i32.gt_u
    (local.get $2)
    (i32.sub
     (i32.load
      (local.get $0)
     )
     (local.tee $3
      (i32.load offset=8
       (local.get $0)
      )
     )
    )
   )
   (then
    (call $12
     (local.get $0)
     (local.get $3)
     (local.get $2)
    )
    (local.set $3
     (i32.load offset=8
      (local.get $0)
     )
    )
   )
  )
  (drop
   (call $2
    (i32.add
     (i32.load offset=4
      (local.get $0)
     )
     (local.get $3)
    )
    (local.get $1)
    (local.get $2)
   )
  )
  (i32.store offset=8
   (local.get $0)
   (i32.add
    (local.get $2)
    (local.get $3)
   )
  )
  (i32.const 0)
 )
 (func $21 (param $0 i32) (param $1 i32)
  (local $2 i32)
  (local $3 i32)
  (drop
   (i32.load8_u
    (i32.const 1050997)
   )
  )
  (local.set $2
   (i32.load offset=4
    (local.get $1)
   )
  )
  (local.set $3
   (i32.load
    (local.get $1)
   )
  )
  (if
   (i32.eqz
    (local.tee $1
     (call $37
      (i32.const 8)
      (i32.const 4)
     )
    )
   )
   (then
    (call $46
     (i32.const 4)
     (i32.const 8)
    )
    (unreachable)
   )
  )
  (i32.store offset=4
   (local.get $1)
   (local.get $2)
  )
  (i32.store
   (local.get $1)
   (local.get $3)
  )
  (i32.store offset=4
   (local.get $0)
   (i32.const 1050132)
  )
  (i32.store
   (local.get $0)
   (local.get $1)
  )
 )
 (func $22 (param $0 i32) (param $1 i32) (param $2 i32)
  (local $3 i32)
  (global.set $global$0
   (local.tee $3
    (i32.sub
     (global.get $global$0)
     (i32.const 32)
    )
   )
  )
  (i32.store offset=16
   (local.get $3)
   (i32.const 0)
  )
  (i32.store offset=4
   (local.get $3)
   (i32.const 1)
  )
  (i64.store offset=8 align=4
   (local.get $3)
   (i64.const 4)
  )
  (i32.store offset=28
   (local.get $3)
   (local.get $1)
  )
  (i32.store offset=24
   (local.get $3)
   (local.get $0)
  )
  (i32.store
   (local.get $3)
   (i32.add
    (local.get $3)
    (i32.const 24)
   )
  )
  (call $26
   (local.get $3)
   (local.get $2)
  )
  (unreachable)
 )
 (func $23 (param $0 i32) (param $1 i32) (param $2 i32) (param $3 i32) (result i32)
  (block $block
   (br_if $block
    (i32.eq
     (local.get $2)
     (i32.const 1114112)
    )
   )
   (br_if $block
    (i32.eqz
     (call_indirect $0 (type $1)
      (local.get $0)
      (local.get $2)
      (i32.load offset=16
       (local.get $1)
      )
     )
    )
   )
   (return
    (i32.const 1)
   )
  )
  (if
   (i32.eqz
    (local.get $3)
   )
   (then
    (return
     (i32.const 0)
    )
   )
  )
  (call_indirect $0 (type $2)
   (local.get $0)
   (local.get $3)
   (i32.const 0)
   (i32.load offset=12
    (local.get $1)
   )
  )
 )
 (func $24 (param $0 i32) (param $1 i32) (result i32)
  (block $block
   (br_if $block
    (i32.eqz
     (i32.and
      (i32.eq
       (i32.popcnt
        (local.get $1)
       )
       (i32.const 1)
      )
      (i32.ge_u
       (i32.sub
        (i32.const -2147483648)
        (local.get $1)
       )
       (local.get $0)
      )
     )
    )
   )
   (if
    (local.get $0)
    (then
     (drop
      (i32.load8_u
       (i32.const 1050997)
      )
     )
     (br_if $block
      (i32.eqz
       (local.tee $1
        (call $37
         (local.get $0)
         (local.get $1)
        )
       )
      )
     )
    )
   )
   (return
    (local.get $1)
   )
  )
  (unreachable)
 )
 (func $25 (param $0 i32) (param $1 i32) (param $2 i32) (param $3 i32) (param $4 i32) (result i32 i32)
  (local $5 f64)
  (local $6 f64)
  (local $7 f64)
  (local $8 f64)
  (local $9 f64)
  (local $10 f64)
  (local $11 f64)
  (local $12 f64)
  (local $13 f64)
  (local $14 f64)
  (local $15 i32)
  (local $16 i32)
  (local $17 i32)
  (local $18 i32)
  (local $19 i32)
  (local $20 i32)
  (local $21 i32)
  (local $22 i32)
  (local $23 i32)
  (local $24 i32)
  (local $25 i32)
  (local $26 i32)
  (local $27 i32)
  (local $28 i32)
  (local $29 i32)
  (local $30 i32)
  (local $31 i32)
  (local $32 i32)
  (local $33 i32)
  (local $34 i32)
  (local $35 i32)
  (local $36 i32)
  (local $37 i32)
  (local $38 i32)
  (local $39 i32)
  (local $40 i32)
  (local $41 i32)
  (local $42 i32)
  (local $43 i32)
  (local $44 i32)
  (local $45 i32)
  (local $46 i32)
  (local $47 i32)
  (local $48 i32)
  (local $49 i32)
  (local $50 i32)
  (local $51 i32)
  (local $52 i32)
  (local $53 i32)
  (local $54 i32)
  (local $55 i32)
  (local $56 i32)
  (local $57 i32)
  (local $58 i32)
  (local $59 i32)
  (local $60 i32)
  (local $61 i32)
  (local $62 i32)
  (local $63 i32)
  (local $64 i32)
  (local $65 i32)
  (local $66 i32)
  (local $67 i32)
  (local $68 i32)
  (local $69 i64)
  (local $70 i64)
  (local $71 v128)
  (local $72 v128)
  (local $73 v128)
  (local $74 v128)
  (local $75 v128)
  (local $scratch i32)
  (local $scratch_77 i32)
  (global.set $global$0
   (local.tee $36
    (i32.sub
     (global.get $global$0)
     (i32.const 16)
    )
   )
  )
  (global.set $global$0
   (local.tee $34
    (i32.sub
     (global.get $global$0)
     (i32.const 16)
    )
   )
  )
  (local.set $50
   (i32.add
    (local.get $34)
    (i32.const 4)
   )
  )
  (global.set $global$0
   (local.tee $18
    (i32.sub
     (global.get $global$0)
     (i32.const 80)
    )
   )
  )
  (i32.store
   (local.get $18)
   (local.tee $25
    (local.get $0)
   )
  )
  (i32.store offset=4
   (local.get $18)
   (local.get $2)
  )
  (block $block41
   (block $block40
    (block $block15
     (block $block19
      (block $block17
       (block $block16
        (block $block1
         (block $block
          (if
           (local.get $4)
           (then
            (br_if $block
             (i32.gt_u
              (local.get $0)
              (local.get $2)
             )
            )
            (local.set $0
             (i32.shl
              (local.tee $31
               (i32.mul
                (i32.add
                 (local.tee $37
                  (i32.div_u
                   (local.tee $40
                    (i32.sub
                     (local.get $2)
                     (local.get $0)
                    )
                   )
                   (local.get $4)
                  )
                 )
                 (i32.const 1)
                )
                (i32.add
                 (i32.shr_u
                  (local.get $0)
                  (i32.const 1)
                 )
                 (i32.const 1)
                )
               )
              )
              (i32.const 3)
             )
            )
            (br_if $block1
             (i32.or
              (i32.gt_u
               (local.get $31)
               (i32.const 536870911)
              )
              (i32.gt_u
               (local.get $0)
               (i32.const 2147483640)
              )
             )
            )
            (block $block2
             (if
              (i32.eqz
               (local.get $0)
              )
              (then
               (local.set $32
                (i32.const 8)
               )
               (local.set $31
                (i32.const 0)
               )
               (br $block2)
              )
             )
             (drop
              (i32.load8_u
               (i32.const 1050997)
              )
             )
             (local.set $16
              (i32.const 8)
             )
             (br_if $block1
              (i32.eqz
               (local.tee $32
                (call $37
                 (local.get $0)
                 (i32.const 8)
                )
               )
              )
             )
            )
            (i32.store offset=28
             (local.get $18)
             (local.get $32)
            )
            (i32.store offset=24
             (local.get $18)
             (local.get $31)
            )
            (i32.store offset=32
             (local.get $18)
             (i32.const 0)
            )
            (local.set $21
             (i32.add
              (local.get $18)
              (i32.const 36)
             )
            )
            (global.set $global$0
             (local.tee $15
              (i32.sub
               (global.get $global$0)
               (i32.const 48)
              )
             )
            )
            (block $block14
             (if
              (i32.eq
               (i32.popcnt
                (local.get $25)
               )
               (i32.const 1)
              )
              (then
               (local.set $20
                (block $block7 (result i32)
                 (if
                  (i32.and
                   (f64.lt
                    (local.tee $5
                     (block $block5 (result f64)
                      (block $block3
                       (block $block6
                        (local.set $0
                         (block $block4 (result i32)
                          (if
                           (i64.ge_s
                            (local.tee $69
                             (i64.reinterpret_f64
                              (local.tee $5
                               (local.tee $8
                                (f64.convert_i32_u
                                 (local.get $25)
                                )
                               )
                              )
                             )
                            )
                            (i64.const 4503599627370496)
                           )
                           (then
                            (br_if $block3
                             (i64.gt_u
                              (local.get $69)
                              (i64.const 9218868437227405311)
                             )
                            )
                            (local.set $16
                             (i32.const -1023)
                            )
                            (if
                             (i64.ne
                              (local.tee $70
                               (i64.shr_u
                                (local.get $69)
                                (i64.const 32)
                               )
                              )
                              (i64.const 1072693248)
                             )
                             (then
                              (br $block4
                               (i32.wrap_i64
                                (local.get $70)
                               )
                              )
                             )
                            )
                            (drop
                             (br_if $block4
                              (i32.const 1072693248)
                              (i32.wrap_i64
                               (local.get $69)
                              )
                             )
                            )
                            (br $block5
                             (f64.const 0)
                            )
                           )
                          )
                          (drop
                           (br_if $block5
                            (f64.div
                             (f64.const -1)
                             (f64.mul
                              (local.get $5)
                              (local.get $5)
                             )
                            )
                            (f64.eq
                             (local.get $5)
                             (f64.const 0)
                            )
                           )
                          )
                          (br_if $block6
                           (i64.lt_s
                            (local.get $69)
                            (i64.const 0)
                           )
                          )
                          (local.set $16
                           (i32.const -1077)
                          )
                          (i32.wrap_i64
                           (i64.shr_u
                            (local.tee $69
                             (i64.reinterpret_f64
                              (f64.mul
                               (local.get $5)
                               (f64.const 18014398509481984)
                              )
                             )
                            )
                            (i64.const 32)
                           )
                          )
                         )
                        )
                        (br $block5
                         (f64.add
                          (local.tee $12
                           (f64.add
                            (local.tee $10
                             (f64.mul
                              (local.tee $9
                               (f64.reinterpret_i64
                                (i64.and
                                 (i64.reinterpret_f64
                                  (f64.sub
                                   (local.tee $5
                                    (f64.add
                                     (f64.reinterpret_i64
                                      (i64.or
                                       (i64.and
                                        (local.get $69)
                                        (i64.const 4294967295)
                                       )
                                       (i64.shl
                                        (i64.extend_i32_u
                                         (i32.add
                                          (i32.and
                                           (local.tee $0
                                            (i32.add
                                             (local.get $0)
                                             (i32.const 614242)
                                            )
                                           )
                                           (i32.const 1048575)
                                          )
                                          (i32.const 1072079006)
                                         )
                                        )
                                        (i64.const 32)
                                       )
                                      )
                                     )
                                     (f64.const -1)
                                    )
                                   )
                                   (local.tee $6
                                    (f64.mul
                                     (local.get $5)
                                     (f64.mul
                                      (local.get $5)
                                      (f64.const 0.5)
                                     )
                                    )
                                   )
                                  )
                                 )
                                 (i64.const -4294967296)
                                )
                               )
                              )
                              (f64.const 1.4426950407214463)
                             )
                            )
                            (local.tee $11
                             (f64.convert_i32_s
                              (i32.add
                               (i32.shr_u
                                (local.get $0)
                                (i32.const 20)
                               )
                               (local.get $16)
                              )
                             )
                            )
                           )
                          )
                          (f64.add
                           (f64.add
                            (local.get $10)
                            (f64.sub
                             (local.get $11)
                             (local.get $12)
                            )
                           )
                           (f64.add
                            (f64.mul
                             (local.tee $5
                              (f64.add
                               (f64.sub
                                (f64.sub
                                 (local.get $5)
                                 (local.get $9)
                                )
                                (local.get $6)
                               )
                               (f64.mul
                                (local.tee $5
                                 (f64.div
                                  (local.get $5)
                                  (f64.add
                                   (local.get $5)
                                   (f64.const 2)
                                  )
                                 )
                                )
                                (f64.add
                                 (local.get $6)
                                 (f64.add
                                  (f64.mul
                                   (local.tee $5
                                    (f64.mul
                                     (local.tee $6
                                      (f64.mul
                                       (local.get $5)
                                       (local.get $5)
                                      )
                                     )
                                     (local.get $6)
                                    )
                                   )
                                   (f64.add
                                    (f64.mul
                                     (local.get $5)
                                     (f64.add
                                      (f64.mul
                                       (local.get $5)
                                       (f64.const 0.15313837699209373)
                                      )
                                      (f64.const 0.22222198432149784)
                                     )
                                    )
                                    (f64.const 0.3999999999940942)
                                   )
                                  )
                                  (f64.mul
                                   (local.get $6)
                                   (f64.add
                                    (f64.mul
                                     (local.get $5)
                                     (f64.add
                                      (f64.mul
                                       (local.get $5)
                                       (f64.add
                                        (f64.mul
                                         (local.get $5)
                                         (f64.const 0.14798198605116586)
                                        )
                                        (f64.const 0.1818357216161805)
                                       )
                                      )
                                      (f64.const 0.2857142874366239)
                                     )
                                    )
                                    (f64.const 0.6666666666666735)
                                   )
                                  )
                                 )
                                )
                               )
                              )
                             )
                             (f64.const 1.4426950407214463)
                            )
                            (f64.mul
                             (f64.add
                              (local.get $5)
                              (local.get $9)
                             )
                             (f64.const 1.6751713164886512e-10)
                            )
                           )
                          )
                         )
                        )
                       )
                       (local.set $5
                        (f64.div
                         (f64.sub
                          (local.get $5)
                          (local.get $5)
                         )
                         (f64.const 0)
                        )
                       )
                      )
                      (local.get $5)
                     )
                    )
                    (f64.const 4294967296)
                   )
                   (local.tee $16
                    (f64.ge
                     (local.get $5)
                     (f64.const 0)
                    )
                   )
                  )
                  (then
                   (br $block7
                    (i32.trunc_f64_u
                     (local.get $5)
                    )
                   )
                  )
                 )
                 (i32.const 0)
                )
               )
               (block $block8
                (br_if $block8
                 (i32.or
                  (i32.gt_u
                   (local.get $25)
                   (i32.const 268435455)
                  )
                  (i32.gt_u
                   (local.tee $0
                    (i32.shl
                     (local.get $25)
                     (i32.const 4)
                    )
                   )
                   (i32.const 2147483640)
                  )
                 )
                )
                (local.set $27
                 (select
                  (i32.const -1)
                  (select
                   (local.get $20)
                   (i32.const 0)
                   (local.get $16)
                  )
                  (f64.gt
                   (local.get $5)
                   (f64.const 4294967295)
                  )
                 )
                )
                (block $block12
                 (block $block9
                  (block $block13
                   (local.set $16
                    (block $block10 (result i32)
                     (if
                      (local.get $0)
                      (then
                       (drop
                        (i32.load8_u
                         (i32.const 1050997)
                        )
                       )
                       (local.set $26
                        (i32.const 8)
                       )
                       (br_if $block8
                        (i32.eqz
                         (local.tee $16
                          (call $37
                           (local.get $0)
                           (i32.const 8)
                          )
                         )
                        )
                       )
                       (i32.store offset=44
                        (local.get $15)
                        (i32.const 0)
                       )
                       (i32.store offset=40
                        (local.get $15)
                        (local.get $16)
                       )
                       (i32.store offset=36
                        (local.get $15)
                        (local.get $25)
                       )
                       (drop
                        (i32.load8_u
                         (i32.const 1050997)
                        )
                       )
                       (br_if $block9
                        (i32.eqz
                         (local.tee $26
                          (call $37
                           (local.get $0)
                           (i32.const 8)
                          )
                         )
                        )
                       )
                       (i32.store offset=20
                        (local.get $15)
                        (i32.const 0)
                       )
                       (i32.store offset=16
                        (local.get $15)
                        (local.get $26)
                       )
                       (i32.store offset=12
                        (local.get $15)
                        (local.get $25)
                       )
                       (drop
                        (br_if $block10
                         (i32.const 0)
                         (i32.eqz
                          (local.get $25)
                         )
                        )
                       )
                       (local.set $5
                        (f64.div
                         (f64.const 6.283185307179586)
                         (local.get $8)
                        )
                       )
                       (local.set $8
                        (f64.div
                         (f64.const -6.283185307179586)
                         (local.get $8)
                        )
                       )
                       (local.set $16
                        (i32.const 0)
                       )
                       (local.set $24
                        (i32.const 8)
                       )
                       (loop $label
                        (local.set $20
                         (i32.load offset=36
                          (local.get $15)
                         )
                        )
                        (local.set $0
                         (i32.load offset=44
                          (local.get $15)
                         )
                        )
                        (local.set $9
                         (call $48
                          (local.tee $6
                           (f64.mul
                            (local.get $8)
                            (local.get $7)
                           )
                          )
                         )
                        )
                        (local.set $6
                         (call $49
                          (local.get $6)
                         )
                        )
                        (if
                         (i32.eq
                          (local.get $0)
                          (local.get $20)
                         )
                         (then
                          (call $13
                           (i32.add
                            (local.get $15)
                            (i32.const 36)
                           )
                           (i32.const 1048724)
                          )
                         )
                        )
                        (f64.store offset=8
                         (local.tee $20
                          (i32.add
                           (i32.load offset=40
                            (local.get $15)
                           )
                           (i32.shl
                            (local.get $0)
                            (i32.const 4)
                           )
                          )
                         )
                         (local.get $9)
                        )
                        (f64.store
                         (local.get $20)
                         (local.get $6)
                        )
                        (i32.store offset=44
                         (local.get $15)
                         (i32.add
                          (local.get $0)
                          (i32.const 1)
                         )
                        )
                        (if
                         (i32.eq
                          (block (result i32)
                           (local.set $scratch
                            (i32.load offset=12
                             (local.get $15)
                            )
                           )
                           (local.set $9
                            (call $48
                             (local.tee $6
                              (f64.mul
                               (local.get $5)
                               (local.get $7)
                              )
                             )
                            )
                           )
                           (local.set $6
                            (call $49
                             (local.get $6)
                            )
                           )
                           (local.set $7
                            (f64.add
                             (local.get $7)
                             (f64.const 1)
                            )
                           )
                           (local.get $scratch)
                          )
                          (local.get $16)
                         )
                         (then
                          (call $13
                           (i32.add
                            (local.get $15)
                            (i32.const 12)
                           )
                           (i32.const 1048740)
                          )
                          (local.set $26
                           (i32.load offset=16
                            (local.get $15)
                           )
                          )
                         )
                        )
                        (f64.store
                         (local.tee $0
                          (i32.add
                           (local.get $24)
                           (local.get $26)
                          )
                         )
                         (local.get $9)
                        )
                        (f64.store
                         (i32.sub
                          (local.get $0)
                          (i32.const 8)
                         )
                         (local.get $6)
                        )
                        (i32.store offset=20
                         (local.get $15)
                         (local.tee $16
                          (i32.add
                           (local.get $16)
                           (i32.const 1)
                          )
                         )
                        )
                        (local.set $24
                         (i32.add
                          (local.get $24)
                          (i32.const 16)
                         )
                        )
                        (br_if $label
                         (i32.ne
                          (local.get $16)
                          (local.get $25)
                         )
                        )
                       )
                       (drop
                        (i32.load8_u
                         (i32.const 1050997)
                        )
                       )
                       (block $block11
                        (br_if $block11
                         (i32.eqz
                          (local.tee $20
                           (call $0
                            (local.tee $0
                             (i32.shl
                              (local.get $25)
                              (i32.const 2)
                             )
                            )
                           )
                          )
                         )
                        )
                        (br_if $block11
                         (i32.eqz
                          (i32.and
                           (i32.load8_u
                            (i32.sub
                             (local.get $20)
                             (i32.const 4)
                            )
                           )
                           (i32.const 3)
                          )
                         )
                        )
                        (call $6
                         (local.get $20)
                         (local.get $0)
                        )
                       )
                       (br_if $block12
                        (i32.eqz
                         (local.get $20)
                        )
                       )
                       (if
                        (local.get $27)
                        (then
                         (local.set $22
                          (i32.and
                           (local.get $27)
                           (i32.const -4)
                          )
                         )
                         (local.set $0
                          (i32.and
                           (local.get $27)
                           (i32.const 3)
                          )
                         )
                         (local.set $28
                          (i32.lt_u
                           (local.get $27)
                           (i32.const 4)
                          )
                         )
                         (loop $label3
                          (local.set $24
                           (i32.const 0)
                          )
                          (local.set $16
                           (i32.const 0)
                          )
                          (if
                           (i32.eqz
                            (local.get $28)
                           )
                           (then
                            (loop $label1
                             (local.set $24
                              (i32.or
                               (i32.and
                                (i32.shr_u
                                 (local.get $23)
                                 (i32.or
                                  (local.tee $26
                                   (i32.and
                                    (local.get $16)
                                    (i32.const 28)
                                   )
                                  )
                                  (i32.const 3)
                                 )
                                )
                                (i32.const 1)
                               )
                               (i32.or
                                (i32.and
                                 (i32.shl
                                  (i32.shr_u
                                   (local.get $23)
                                   (i32.or
                                    (local.get $26)
                                    (i32.const 2)
                                   )
                                  )
                                  (i32.const 1)
                                 )
                                 (i32.const 2)
                                )
                                (i32.shl
                                 (i32.or
                                  (i32.and
                                   (i32.shr_u
                                    (local.get $23)
                                    (i32.or
                                     (local.get $26)
                                     (i32.const 1)
                                    )
                                   )
                                   (i32.const 1)
                                  )
                                  (i32.or
                                   (i32.and
                                    (i32.shl
                                     (i32.shr_u
                                      (local.get $23)
                                      (local.get $26)
                                     )
                                     (i32.const 1)
                                    )
                                    (i32.const 2)
                                   )
                                   (i32.shl
                                    (local.get $24)
                                    (i32.const 2)
                                   )
                                  )
                                 )
                                 (i32.const 2)
                                )
                               )
                              )
                             )
                             (br_if $label1
                              (i32.ne
                               (local.get $22)
                               (local.tee $16
                                (i32.add
                                 (local.get $16)
                                 (i32.const 4)
                                )
                               )
                              )
                             )
                            )
                           )
                          )
                          (if
                           (local.get $0)
                           (then
                            (local.set $26
                             (local.get $0)
                            )
                            (loop $label2
                             (local.set $24
                              (i32.or
                               (i32.and
                                (i32.shr_u
                                 (local.get $23)
                                 (local.get $16)
                                )
                                (i32.const 1)
                               )
                               (i32.shl
                                (local.get $24)
                                (i32.const 1)
                               )
                              )
                             )
                             (local.set $16
                              (i32.add
                               (local.get $16)
                               (i32.const 1)
                              )
                             )
                             (br_if $label2
                              (local.tee $26
                               (i32.sub
                                (local.get $26)
                                (i32.const 1)
                               )
                              )
                             )
                            )
                           )
                          )
                          (i32.store
                           (i32.add
                            (local.get $20)
                            (i32.shl
                             (local.get $23)
                             (i32.const 2)
                            )
                           )
                           (local.get $24)
                          )
                          (br_if $label3
                           (i32.ne
                            (local.tee $23
                             (i32.add
                              (local.get $23)
                              (i32.const 1)
                             )
                            )
                            (local.get $25)
                           )
                          )
                         )
                         (local.set $16
                          (local.get $25)
                         )
                         (br $block13)
                        )
                       )
                       (call $6
                        (local.get $20)
                        (local.get $0)
                       )
                       (local.set $16
                        (local.get $25)
                       )
                       (br $block13)
                      )
                     )
                     (i32.store offset=44
                      (local.get $15)
                      (i32.const 0)
                     )
                     (i64.store offset=36 align=4
                      (local.get $15)
                      (i64.const 34359738368)
                     )
                     (i32.store offset=20
                      (local.get $15)
                      (i32.const 0)
                     )
                     (i64.store offset=12 align=4
                      (local.get $15)
                      (i64.const 34359738368)
                     )
                     (i32.const 0)
                    )
                   )
                   (local.set $20
                    (i32.const 4)
                   )
                  )
                  (i64.store align=4
                   (local.get $21)
                   (i64.load offset=36 align=4
                    (local.get $15)
                   )
                  )
                  (i64.store offset=12 align=4
                   (local.get $21)
                   (i64.load offset=12 align=4
                    (local.get $15)
                   )
                  )
                  (i32.store offset=40
                   (local.get $21)
                   (local.get $27)
                  )
                  (i32.store offset=36
                   (local.get $21)
                   (local.get $25)
                  )
                  (i32.store offset=32
                   (local.get $21)
                   (local.get $25)
                  )
                  (i32.store offset=28
                   (local.get $21)
                   (local.get $20)
                  )
                  (i32.store offset=24
                   (local.get $21)
                   (local.get $16)
                  )
                  (i32.store
                   (i32.add
                    (local.get $21)
                    (i32.const 8)
                   )
                   (i32.load
                    (i32.add
                     (local.get $15)
                     (i32.const 44)
                    )
                   )
                  )
                  (i32.store
                   (i32.add
                    (local.get $21)
                    (i32.const 20)
                   )
                   (i32.load
                    (i32.add
                     (local.get $15)
                     (i32.const 20)
                    )
                   )
                  )
                  (global.set $global$0
                   (i32.add
                    (local.get $15)
                    (i32.const 48)
                   )
                  )
                  (br $block14)
                 )
                 (call $29
                  (i32.const 8)
                  (local.get $0)
                  (i32.const 1048692)
                 )
                 (unreachable)
                )
                (call $29
                 (i32.const 4)
                 (local.get $0)
                 (i32.const 1048708)
                )
                (unreachable)
               )
               (call $29
                (local.get $26)
                (local.get $0)
                (i32.const 1048676)
               )
               (unreachable)
              )
             )
             (i32.store offset=28
              (local.get $15)
              (i32.const 0)
             )
             (i32.store offset=16
              (local.get $15)
              (i32.const 1)
             )
             (i32.store offset=12
              (local.get $15)
              (i32.const 1048780)
             )
             (i64.store offset=20 align=4
              (local.get $15)
              (i64.const 4)
             )
             (call $26
              (i32.add
               (local.get $15)
               (i32.const 12)
              )
              (i32.const 1048788)
             )
             (unreachable)
            )
            (br_if $block15
             (i32.eqz
              (local.tee $44
               (i32.add
                (local.get $37)
                (i32.ne
                 (local.get $40)
                 (i32.mul
                  (local.get $4)
                  (local.get $37)
                 )
                )
               )
              )
             )
            )
            (local.set $56
             (i32.shl
              (local.get $4)
              (i32.const 3)
             )
            )
            (local.set $57
             (i32.and
              (local.get $25)
              (i32.const -2)
             )
            )
            (local.set $58
             (i32.and
              (local.get $25)
              (i32.const 1)
             )
            )
            (local.set $45
             (i32.shl
              (local.get $25)
              (i32.const 3)
             )
            )
            (local.set $26
             (local.get $1)
            )
            (local.set $31
             (i32.const 0)
            )
            (local.set $16
             (local.get $4)
            )
            (loop $label12
             (br_if $block16
              (i32.lt_u
               (local.tee $15
                (i32.add
                 (local.get $19)
                 (local.get $25)
                )
               )
               (local.get $19)
              )
             )
             (br_if $block17
              (i32.lt_u
               (local.get $2)
               (local.get $15)
              )
             )
             (local.set $37
              (local.get $16)
             )
             (block $block18
              (if
               (i32.eqz
                (local.get $25)
               )
               (then
                (local.set $38
                 (i32.const 8)
                )
                (br $block18)
               )
              )
              (drop
               (i32.load8_u
                (i32.const 1050997)
               )
              )
              (br_if $block19
               (i32.eqz
                (local.tee $38
                 (call $37
                  (local.get $45)
                  (i32.const 8)
                 )
                )
               )
              )
              (local.set $16
               (i32.const 0)
              )
              (if
               (i32.ne
                (local.get $25)
                (i32.const 1)
               )
               (then
                (local.set $0
                 (local.get $38)
                )
                (local.set $15
                 (local.get $26)
                )
                (loop $label4
                 (f64.store
                  (local.get $0)
                  (f64.mul
                   (f64.load
                    (local.get $15)
                   )
                   (f64.load
                    (i32.add
                     (local.get $3)
                     (i32.shl
                      (i32.rem_u
                       (local.get $16)
                       (local.get $4)
                      )
                      (i32.const 3)
                     )
                    )
                   )
                  )
                 )
                 (f64.store
                  (i32.add
                   (local.get $0)
                   (i32.const 8)
                  )
                  (f64.mul
                   (f64.load
                    (i32.add
                     (local.get $15)
                     (i32.const 8)
                    )
                   )
                   (f64.load
                    (i32.add
                     (local.get $3)
                     (i32.shl
                      (i32.rem_u
                       (i32.add
                        (local.get $16)
                        (i32.const 1)
                       )
                       (local.get $4)
                      )
                      (i32.const 3)
                     )
                    )
                   )
                  )
                 )
                 (local.set $0
                  (i32.add
                   (local.get $0)
                   (i32.const 16)
                  )
                 )
                 (local.set $15
                  (i32.add
                   (local.get $15)
                   (i32.const 16)
                  )
                 )
                 (br_if $label4
                  (i32.ne
                   (local.get $57)
                   (local.tee $16
                    (i32.add
                     (local.get $16)
                     (i32.const 2)
                    )
                   )
                  )
                 )
                )
               )
              )
              (br_if $block18
               (i32.eqz
                (local.get $58)
               )
              )
              (f64.store
               (i32.add
                (local.get $38)
                (local.tee $0
                 (i32.shl
                  (local.get $16)
                  (i32.const 3)
                 )
                )
               )
               (f64.mul
                (f64.load
                 (i32.add
                  (i32.add
                   (local.get $1)
                   (i32.shl
                    (local.get $19)
                    (i32.const 3)
                   )
                  )
                  (local.get $0)
                 )
                )
                (f64.load
                 (i32.add
                  (local.get $3)
                  (i32.shl
                   (i32.rem_u
                    (local.get $16)
                    (local.get $4)
                   )
                   (i32.const 3)
                  )
                 )
                )
               )
              )
             )
             (local.set $51
              (i32.add
               (local.get $18)
               (i32.const 8)
              )
             )
             (local.set $0
              (i32.const 0)
             )
             (local.set $19
              (i32.const 0)
             )
             (local.set $29
              (i32.const 0)
             )
             (global.set $global$0
              (local.tee $17
               (i32.sub
                (global.get $global$0)
                (i32.const 288)
               )
              )
             )
             (local.set $15
              (i32.shl
               (local.tee $27
                (i32.load offset=36
                 (local.tee $16
                  (i32.add
                   (local.get $18)
                   (i32.const 36)
                  )
                 )
                )
               )
               (i32.const 4)
              )
             )
             (block $block34
              (block $block20
               (br_if $block20
                (i32.or
                 (i32.gt_u
                  (local.get $27)
                  (i32.const 268435455)
                 )
                 (i32.gt_u
                  (local.get $15)
                  (i32.const 2147483640)
                 )
                )
               )
               (local.set $33
                (i32.load offset=8
                 (local.get $16)
                )
               )
               (local.set $41
                (i32.load offset=4
                 (local.get $16)
                )
               )
               (block $block21
                (if
                 (i32.eqz
                  (local.get $15)
                 )
                 (then
                  (local.set $23
                   (i32.const 8)
                  )
                  (br $block21)
                 )
                )
                (drop
                 (i32.load8_u
                  (i32.const 1050997)
                 )
                )
                (local.set $29
                 (i32.const 8)
                )
                (local.set $0
                 (local.get $27)
                )
                (br_if $block20
                 (i32.eqz
                  (local.tee $23
                   (call $37
                    (local.get $15)
                    (i32.const 8)
                   )
                  )
                 )
                )
               )
               (i32.store offset=12
                (local.get $17)
                (i32.const 0)
               )
               (i32.store offset=8
                (local.get $17)
                (local.get $23)
               )
               (i32.store offset=4
                (local.get $17)
                (local.get $0)
               )
               (block $block22
                (if
                 (local.tee $0
                  (i32.load offset=32
                   (local.get $16)
                  )
                 )
                 (then
                  (local.set $15
                   (i32.load offset=28
                    (local.get $16)
                   )
                  )
                  (local.set $0
                   (i32.shl
                    (local.get $0)
                    (i32.const 2)
                   )
                  )
                  (local.set $16
                   (i32.const 8)
                  )
                  (loop $label5
                   (br_if $block22
                    (i32.ge_u
                     (local.tee $20
                      (i32.load
                       (local.get $15)
                      )
                     )
                     (local.get $25)
                    )
                   )
                   (local.set $5
                    (f64.load
                     (i32.add
                      (local.get $38)
                      (i32.shl
                       (local.get $20)
                       (i32.const 3)
                      )
                     )
                    )
                   )
                   (local.set $15
                    (i32.add
                     (local.get $15)
                     (i32.const 4)
                    )
                   )
                   (if
                    (i32.eq
                     (i32.load offset=4
                      (local.get $17)
                     )
                     (local.get $19)
                    )
                    (then
                     (call $13
                      (i32.add
                       (local.get $17)
                       (i32.const 4)
                      )
                      (i32.const 1049008)
                     )
                     (local.set $23
                      (i32.load offset=8
                       (local.get $17)
                      )
                     )
                    )
                   )
                   (i64.store
                    (local.tee $20
                     (i32.add
                      (local.get $16)
                      (local.get $23)
                     )
                    )
                    (i64.const 0)
                   )
                   (f64.store
                    (i32.sub
                     (local.get $20)
                     (i32.const 8)
                    )
                    (local.get $5)
                   )
                   (i32.store offset=12
                    (local.get $17)
                    (local.tee $19
                     (i32.add
                      (local.get $19)
                      (i32.const 1)
                     )
                    )
                   )
                   (local.set $16
                    (i32.add
                     (local.get $16)
                     (i32.const 16)
                    )
                   )
                   (br_if $label5
                    (local.tee $0
                     (i32.sub
                      (local.get $0)
                      (i32.const 4)
                     )
                    )
                   )
                  )
                 )
                )
                (if
                 (i32.ge_u
                  (local.get $27)
                  (i32.const 2)
                 )
                 (then
                  (local.set $40
                   (i32.load offset=8
                    (local.get $17)
                   )
                  )
                  (local.set $24
                   (local.get $27)
                  )
                  (local.set $30
                   (i32.const 1)
                  )
                  (loop $label10
                   (local.set $59
                    (local.get $24)
                   )
                   (block $block23
                    (block $block31
                     (block $block33
                      (block $block32
                       (block $block24
                        (if
                         (local.tee $30
                          (i32.shl
                           (local.tee $21
                            (local.get $30)
                           )
                           (i32.const 1)
                          )
                         )
                         (then
                          (local.set $24
                           (i32.shr_u
                            (local.get $24)
                            (i32.const 1)
                           )
                          )
                          (br_if $block23
                           (i32.eqz
                            (local.tee $0
                             (i32.add
                              (i32.div_u
                               (local.get $27)
                               (local.get $30)
                              )
                              (i32.ne
                               (i32.and
                                (i32.add
                                 (local.get $30)
                                 (i32.const 268435455)
                                )
                                (local.get $27)
                               )
                               (i32.const 0)
                              )
                             )
                            )
                           )
                          )
                          (local.set $16
                           (i32.sub
                            (local.get $0)
                            (i32.const 1)
                           )
                          )
                          (br_if $block24
                           (i32.lt_u
                            (local.get $21)
                            (i32.const 2)
                           )
                          )
                          (local.set $60
                           (i32.sub
                            (local.get $21)
                            (i32.const 2)
                           )
                          )
                          (local.set $52
                           (i32.shl
                            (local.get $24)
                            (i32.const 4)
                           )
                          )
                          (local.set $61
                           (i32.shl
                            (local.get $21)
                            (i32.const 5)
                           )
                          )
                          (local.set $53
                           (i32.shl
                            (local.get $21)
                            (i32.const 4)
                           )
                          )
                          (local.set $46
                           (i32.shl
                            (local.get $24)
                            (i32.const 1)
                           )
                          )
                          (local.set $62
                           (i32.add
                            (local.get $41)
                            (local.tee $54
                             (i32.shl
                              (local.get $24)
                              (i32.const 5)
                             )
                            )
                           )
                          )
                          (local.set $47
                           (i32.const 0)
                          )
                          (local.set $48
                           (i32.const 2)
                          )
                          (local.set $20
                           (local.get $40)
                          )
                          (local.set $49
                           (local.get $21)
                          )
                          (block $block30
                           (block $block29
                            (block $block28
                             (block $block27
                              (block $block26
                               (loop $label8
                                (local.set $55
                                 (local.get $16)
                                )
                                (local.set $23
                                 (local.get $48)
                                )
                                (local.set $22
                                 (local.get $62)
                                )
                                (local.set $28
                                 (local.get $46)
                                )
                                (local.set $15
                                 (local.get $20)
                                )
                                (local.set $16
                                 (local.get $41)
                                )
                                (local.set $29
                                 (i32.const 0)
                                )
                                (local.set $0
                                 (i32.const 0)
                                )
                                (block $block25
                                 (loop $label6
                                  (local.set $35
                                   (local.get $28)
                                  )
                                  (local.set $39
                                   (local.get $22)
                                  )
                                  (local.set $42
                                   (local.get $23)
                                  )
                                  (br_if $block25
                                   (i32.ge_u
                                    (local.tee $22
                                     (i32.add
                                      (local.get $47)
                                      (local.tee $43
                                       (local.get $0)
                                      )
                                     )
                                    )
                                    (local.get $19)
                                   )
                                  )
                                  (br_if $block26
                                   (i32.ge_u
                                    (i32.add
                                     (local.get $22)
                                     (i32.const 1)
                                    )
                                    (local.get $19)
                                   )
                                  )
                                  (br_if $block27
                                   (i32.ge_u
                                    (local.tee $22
                                     (i32.add
                                      (local.get $0)
                                      (local.get $49)
                                     )
                                    )
                                    (local.get $19)
                                   )
                                  )
                                  (br_if $block28
                                   (i32.ge_u
                                    (local.get $29)
                                    (local.get $33)
                                   )
                                  )
                                  (br_if $block29
                                   (i32.ge_u
                                    (i32.add
                                     (local.get $22)
                                     (i32.const 1)
                                    )
                                    (local.get $19)
                                   )
                                  )
                                  (br_if $block30
                                   (i32.ge_u
                                    (i32.add
                                     (local.get $24)
                                     (local.get $29)
                                    )
                                    (local.get $33)
                                   )
                                  )
                                  (local.set $13
                                   (f64.load
                                    (local.tee $63
                                     (i32.add
                                      (local.get $15)
                                      (i32.const 8)
                                     )
                                    )
                                   )
                                  )
                                  (local.set $14
                                   (f64.load
                                    (local.tee $64
                                     (i32.add
                                      (local.get $15)
                                      (i32.const 24)
                                     )
                                    )
                                   )
                                  )
                                  (local.set $5
                                   (f64.load
                                    (i32.add
                                     (local.get $16)
                                     (i32.const 8)
                                    )
                                   )
                                  )
                                  (local.set $7
                                   (f64.load
                                    (local.tee $28
                                     (i32.add
                                      (local.get $15)
                                      (local.get $53)
                                     )
                                    )
                                   )
                                  )
                                  (local.set $8
                                   (f64.load
                                    (local.tee $65
                                     (i32.add
                                      (local.get $28)
                                      (i32.const 8)
                                     )
                                    )
                                   )
                                  )
                                  (local.set $6
                                   (f64.load
                                    (local.get $16)
                                   )
                                  )
                                  (local.set $9
                                   (f64.load
                                    (local.tee $22
                                     (i32.add
                                      (local.get $16)
                                      (local.get $52)
                                     )
                                    )
                                   )
                                  )
                                  (local.set $10
                                   (f64.load
                                    (local.tee $66
                                     (i32.add
                                      (local.get $28)
                                      (i32.const 16)
                                     )
                                    )
                                   )
                                  )
                                  (local.set $11
                                   (f64.load
                                    (i32.add
                                     (local.get $22)
                                     (i32.const 8)
                                    )
                                   )
                                  )
                                  (local.set $12
                                   (f64.load
                                    (local.tee $67
                                     (i32.add
                                      (local.get $28)
                                      (i32.const 24)
                                     )
                                    )
                                   )
                                  )
                                  (call $36
                                   (local.tee $22
                                    (i32.add
                                     (local.get $17)
                                     (i32.const 112)
                                    )
                                   )
                                   (f64.load
                                    (local.get $15)
                                   )
                                   (f64.load
                                    (local.tee $68
                                     (i32.add
                                      (local.get $15)
                                      (i32.const 16)
                                     )
                                    )
                                   )
                                  )
                                  (local.set $71
                                   (v128.load offset=112
                                    (local.get $17)
                                   )
                                  )
                                  (call $36
                                   (local.get $22)
                                   (local.get $13)
                                   (local.get $14)
                                  )
                                  (local.set $72
                                   (v128.load offset=112
                                    (local.get $17)
                                   )
                                  )
                                  (call $36
                                   (local.get $22)
                                   (f64.sub
                                    (f64.mul
                                     (local.get $7)
                                     (local.get $6)
                                    )
                                    (f64.mul
                                     (local.get $8)
                                     (local.get $5)
                                    )
                                   )
                                   (f64.sub
                                    (f64.mul
                                     (local.get $10)
                                     (local.get $9)
                                    )
                                    (f64.mul
                                     (local.get $12)
                                     (local.get $11)
                                    )
                                   )
                                  )
                                  (local.set $73
                                   (v128.load offset=112
                                    (local.get $17)
                                   )
                                  )
                                  (call $36
                                   (local.get $22)
                                   (f64.add
                                    (f64.mul
                                     (local.get $8)
                                     (local.get $6)
                                    )
                                    (f64.mul
                                     (local.get $7)
                                     (local.get $5)
                                    )
                                   )
                                   (f64.add
                                    (f64.mul
                                     (local.get $12)
                                     (local.get $9)
                                    )
                                    (f64.mul
                                     (local.get $10)
                                     (local.get $11)
                                    )
                                   )
                                  )
                                  (v128.store offset=16
                                   (local.get $17)
                                   (local.get $71)
                                  )
                                  (v128.store offset=32
                                   (local.get $17)
                                   (local.get $73)
                                  )
                                  (local.set $74
                                   (v128.load offset=112
                                    (local.get $17)
                                   )
                                  )
                                  (call $34
                                   (local.get $22)
                                   (i32.add
                                    (local.get $17)
                                    (i32.const 16)
                                   )
                                   (i32.add
                                    (local.get $17)
                                    (i32.const 32)
                                   )
                                  )
                                  (v128.store offset=48
                                   (local.get $17)
                                   (local.get $72)
                                  )
                                  (v128.store offset=64
                                   (local.get $17)
                                   (local.get $74)
                                  )
                                  (local.set $75
                                   (v128.load offset=112
                                    (local.get $17)
                                   )
                                  )
                                  (call $34
                                   (local.get $22)
                                   (i32.add
                                    (local.get $17)
                                    (i32.const 48)
                                   )
                                   (i32.sub
                                    (local.get $17)
                                    (i32.const -64)
                                   )
                                  )
                                  (v128.store offset=80
                                   (local.get $17)
                                   (local.get $71)
                                  )
                                  (v128.store offset=96
                                   (local.get $17)
                                   (local.get $73)
                                  )
                                  (local.set $71
                                   (v128.load offset=112
                                    (local.get $17)
                                   )
                                  )
                                  (call $35
                                   (local.get $22)
                                   (i32.add
                                    (local.get $17)
                                    (i32.const 80)
                                   )
                                   (i32.add
                                    (local.get $17)
                                    (i32.const 96)
                                   )
                                  )
                                  (v128.store offset=128
                                   (local.get $17)
                                   (local.get $72)
                                  )
                                  (v128.store offset=144
                                   (local.get $17)
                                   (local.get $74)
                                  )
                                  (local.set $72
                                   (v128.load offset=112
                                    (local.get $17)
                                   )
                                  )
                                  (call $35
                                   (local.get $22)
                                   (i32.add
                                    (local.get $17)
                                    (i32.const 128)
                                   )
                                   (i32.add
                                    (local.get $17)
                                    (i32.const 144)
                                   )
                                  )
                                  (v128.store offset=160
                                   (local.get $17)
                                   (local.get $75)
                                  )
                                  (local.set $73
                                   (v128.load offset=112
                                    (local.get $17)
                                   )
                                  )
                                  (local.set $5
                                   (f64.load
                                    (i32.add
                                     (local.get $17)
                                     (i32.const 160)
                                    )
                                   )
                                  )
                                  (v128.store offset=176
                                   (local.get $17)
                                   (local.get $71)
                                  )
                                  (f64.store
                                   (local.get $63)
                                   (f64.load
                                    (i32.add
                                     (local.get $17)
                                     (i32.const 176)
                                    )
                                   )
                                  )
                                  (f64.store
                                   (local.get $15)
                                   (local.get $5)
                                  )
                                  (v128.store offset=192
                                   (local.get $17)
                                   (local.get $75)
                                  )
                                  (local.set $5
                                   (f64.load offset=8
                                    (i32.add
                                     (local.get $17)
                                     (i32.const 192)
                                    )
                                   )
                                  )
                                  (v128.store offset=208
                                   (local.get $17)
                                   (local.get $71)
                                  )
                                  (f64.store
                                   (local.get $64)
                                   (f64.load offset=8
                                    (i32.add
                                     (local.get $17)
                                     (i32.const 208)
                                    )
                                   )
                                  )
                                  (f64.store
                                   (local.get $68)
                                   (local.get $5)
                                  )
                                  (v128.store offset=224
                                   (local.get $17)
                                   (local.get $72)
                                  )
                                  (local.set $5
                                   (f64.load
                                    (i32.add
                                     (local.get $17)
                                     (i32.const 224)
                                    )
                                   )
                                  )
                                  (v128.store offset=240
                                   (local.get $17)
                                   (local.get $73)
                                  )
                                  (f64.store
                                   (local.get $65)
                                   (f64.load
                                    (i32.add
                                     (local.get $17)
                                     (i32.const 240)
                                    )
                                   )
                                  )
                                  (f64.store
                                   (local.get $28)
                                   (local.get $5)
                                  )
                                  (v128.store offset=256
                                   (local.get $17)
                                   (local.get $72)
                                  )
                                  (local.set $5
                                   (f64.load offset=8
                                    (i32.add
                                     (local.get $17)
                                     (i32.const 256)
                                    )
                                   )
                                  )
                                  (v128.store offset=272
                                   (local.get $17)
                                   (local.get $73)
                                  )
                                  (f64.store
                                   (local.get $67)
                                   (f64.load offset=8
                                    (i32.add
                                     (local.get $17)
                                     (i32.const 272)
                                    )
                                   )
                                  )
                                  (f64.store
                                   (local.get $66)
                                   (local.get $5)
                                  )
                                  (local.set $23
                                   (i32.add
                                    (local.get $23)
                                    (i32.const 2)
                                   )
                                  )
                                  (local.set $22
                                   (i32.add
                                    (local.get $39)
                                    (local.get $54)
                                   )
                                  )
                                  (local.set $28
                                   (i32.add
                                    (local.get $35)
                                    (local.get $46)
                                   )
                                  )
                                  (local.set $15
                                   (i32.add
                                    (local.get $15)
                                    (i32.const 32)
                                   )
                                  )
                                  (local.set $16
                                   (i32.add
                                    (local.get $16)
                                    (local.get $54)
                                   )
                                  )
                                  (local.set $29
                                   (i32.add
                                    (local.get $29)
                                    (local.get $46)
                                   )
                                  )
                                  (local.set $0
                                   (i32.add
                                    (local.get $0)
                                    (i32.const 2)
                                   )
                                  )
                                  (br_if $label6
                                   (i32.lt_u
                                    (i32.add
                                     (local.get $43)
                                     (i32.const 3)
                                    )
                                    (local.get $21)
                                   )
                                  )
                                 )
                                 (if
                                  (i32.lt_u
                                   (local.get $0)
                                   (local.get $21)
                                  )
                                  (then
                                   (local.set $23
                                    (i32.sub
                                     (local.get $60)
                                     (local.get $43)
                                    )
                                   )
                                   (local.set $28
                                    (i32.add
                                     (local.get $15)
                                     (local.get $53)
                                    )
                                   )
                                   (local.set $16
                                    (i32.const 0)
                                   )
                                   (loop $label7
                                    (if
                                     (i32.le_u
                                      (local.get $19)
                                      (local.get $42)
                                     )
                                     (then
                                      (local.set $0
                                       (local.get $19)
                                      )
                                      (br $block31)
                                     )
                                    )
                                    (if
                                     (i32.le_u
                                      (local.get $19)
                                      (i32.add
                                       (local.get $21)
                                       (local.get $42)
                                      )
                                     )
                                     (then
                                      (local.set $16
                                       (local.get $19)
                                      )
                                      (br $block32)
                                     )
                                    )
                                    (br_if $block33
                                     (i32.le_u
                                      (local.get $33)
                                      (local.get $35)
                                     )
                                    )
                                    (local.set $5
                                     (f64.load
                                      (local.tee $0
                                       (i32.add
                                        (local.get $15)
                                        (local.get $16)
                                       )
                                      )
                                     )
                                    )
                                    (f64.store
                                     (local.tee $22
                                      (i32.add
                                       (local.get $0)
                                       (i32.const 8)
                                      )
                                     )
                                     (f64.add
                                      (local.tee $7
                                       (f64.load
                                        (local.get $22)
                                       )
                                      )
                                      (local.tee $11
                                       (f64.add
                                        (f64.mul
                                         (local.tee $8
                                          (f64.load
                                           (local.tee $43
                                            (i32.add
                                             (local.tee $22
                                              (i32.add
                                               (local.get $16)
                                               (local.get $28)
                                              )
                                             )
                                             (i32.const 8)
                                            )
                                           )
                                          )
                                         )
                                         (local.tee $6
                                          (f64.load
                                           (local.get $39)
                                          )
                                         )
                                        )
                                        (f64.mul
                                         (local.tee $9
                                          (f64.load
                                           (local.get $22)
                                          )
                                         )
                                         (local.tee $10
                                          (f64.load
                                           (i32.add
                                            (local.get $39)
                                            (i32.const 8)
                                           )
                                          )
                                         )
                                        )
                                       )
                                      )
                                     )
                                    )
                                    (f64.store
                                     (local.get $0)
                                     (f64.add
                                      (local.get $5)
                                      (local.tee $8
                                       (f64.sub
                                        (f64.mul
                                         (local.get $9)
                                         (local.get $6)
                                        )
                                        (f64.mul
                                         (local.get $8)
                                         (local.get $10)
                                        )
                                       )
                                      )
                                     )
                                    )
                                    (f64.store
                                     (local.get $43)
                                     (f64.sub
                                      (local.get $7)
                                      (local.get $11)
                                     )
                                    )
                                    (f64.store
                                     (local.get $22)
                                     (f64.sub
                                      (local.get $5)
                                      (local.get $8)
                                     )
                                    )
                                    (local.set $42
                                     (i32.add
                                      (local.get $42)
                                      (i32.const 1)
                                     )
                                    )
                                    (local.set $16
                                     (i32.add
                                      (local.get $16)
                                      (i32.const 16)
                                     )
                                    )
                                    (local.set $39
                                     (i32.add
                                      (local.get $39)
                                      (local.get $52)
                                     )
                                    )
                                    (local.set $35
                                     (i32.add
                                      (local.get $24)
                                      (local.get $35)
                                     )
                                    )
                                    (br_if $label7
                                     (local.tee $23
                                      (i32.sub
                                       (local.get $23)
                                       (i32.const 1)
                                      )
                                     )
                                    )
                                   )
                                  )
                                 )
                                 (local.set $48
                                  (i32.add
                                   (local.get $30)
                                   (local.get $48)
                                  )
                                 )
                                 (local.set $20
                                  (i32.add
                                   (local.get $20)
                                   (local.get $61)
                                  )
                                 )
                                 (local.set $49
                                  (i32.add
                                   (local.get $30)
                                   (local.get $49)
                                  )
                                 )
                                 (local.set $47
                                  (i32.add
                                   (local.get $30)
                                   (local.get $47)
                                  )
                                 )
                                 (local.set $16
                                  (i32.sub
                                   (local.get $55)
                                   (i32.const 1)
                                  )
                                 )
                                 (br_if $block23
                                  (i32.eqz
                                   (local.get $55)
                                  )
                                 )
                                 (br $label8)
                                )
                               )
                               (call $18
                                (local.get $22)
                                (local.get $19)
                                (i32.const 1048912)
                               )
                               (unreachable)
                              )
                              (call $18
                               (i32.add
                                (local.get $22)
                                (i32.const 1)
                               )
                               (local.get $19)
                               (i32.const 1048928)
                              )
                              (unreachable)
                             )
                             (call $18
                              (local.get $22)
                              (local.get $19)
                              (i32.const 1048944)
                             )
                             (unreachable)
                            )
                            (call $18
                             (local.get $29)
                             (local.get $33)
                             (i32.const 1048960)
                            )
                            (unreachable)
                           )
                           (call $18
                            (i32.add
                             (local.get $22)
                             (i32.const 1)
                            )
                            (local.get $19)
                            (i32.const 1048976)
                           )
                           (unreachable)
                          )
                          (call $18
                           (i32.add
                            (i32.shr_u
                             (local.get $59)
                             (i32.const 1)
                            )
                            (local.get $29)
                           )
                           (local.get $33)
                           (i32.const 1048992)
                          )
                          (unreachable)
                         )
                        )
                        (call $22
                         (i32.const 1048820)
                         (i32.const 27)
                         (i32.const 1048848)
                        )
                        (unreachable)
                       )
                       (if
                        (local.get $33)
                        (then
                         (local.set $23
                          (i32.shl
                           (local.get $21)
                           (i32.const 4)
                          )
                         )
                         (local.set $22
                          (i32.shl
                           (local.get $21)
                           (i32.const 5)
                          )
                         )
                         (local.set $0
                          (i32.const 0)
                         )
                         (local.set $15
                          (local.get $40)
                         )
                         (loop $label9
                          (br_if $block31
                           (i32.ge_u
                            (local.get $0)
                            (local.get $19)
                           )
                          )
                          (local.set $20
                           (local.get $16)
                          )
                          (br_if $block32
                           (i32.ge_u
                            (local.tee $16
                             (i32.add
                              (local.get $0)
                              (local.get $21)
                             )
                            )
                            (local.get $19)
                           )
                          )
                          (f64.store
                           (local.tee $16
                            (i32.add
                             (local.get $15)
                             (i32.const 8)
                            )
                           )
                           (f64.add
                            (local.tee $5
                             (f64.load
                              (local.get $16)
                             )
                            )
                            (local.tee $10
                             (f64.add
                              (f64.mul
                               (local.tee $7
                                (f64.load
                                 (local.tee $28
                                  (i32.add
                                   (local.tee $16
                                    (i32.add
                                     (local.get $15)
                                     (local.get $23)
                                    )
                                   )
                                   (i32.const 8)
                                  )
                                 )
                                )
                               )
                               (local.tee $8
                                (f64.load
                                 (local.get $41)
                                )
                               )
                              )
                              (f64.mul
                               (local.tee $6
                                (f64.load
                                 (local.get $16)
                                )
                               )
                               (local.tee $9
                                (f64.load offset=8
                                 (local.get $41)
                                )
                               )
                              )
                             )
                            )
                           )
                          )
                          (f64.store
                           (local.get $15)
                           (f64.add
                            (local.tee $11
                             (f64.load
                              (local.get $15)
                             )
                            )
                            (local.tee $7
                             (f64.sub
                              (f64.mul
                               (local.get $6)
                               (local.get $8)
                              )
                              (f64.mul
                               (local.get $7)
                               (local.get $9)
                              )
                             )
                            )
                           )
                          )
                          (f64.store
                           (local.get $28)
                           (f64.sub
                            (local.get $5)
                            (local.get $10)
                           )
                          )
                          (f64.store
                           (local.get $16)
                           (f64.sub
                            (local.get $11)
                            (local.get $7)
                           )
                          )
                          (local.set $15
                           (i32.add
                            (local.get $15)
                            (local.get $22)
                           )
                          )
                          (local.set $0
                           (i32.add
                            (local.get $0)
                            (local.get $30)
                           )
                          )
                          (local.set $16
                           (i32.sub
                            (local.get $20)
                            (i32.const 1)
                           )
                          )
                          (br_if $label9
                           (local.get $20)
                          )
                         )
                         (br $block23)
                        )
                       )
                       (local.set $0
                        (i32.const 0)
                       )
                       (br_if $block31
                        (i32.eqz
                         (local.get $19)
                        )
                       )
                       (local.set $16
                        (i32.const 1)
                       )
                       (local.set $35
                        (i32.const 0)
                       )
                       (br_if $block33
                        (i32.gt_u
                         (local.get $19)
                         (local.get $21)
                        )
                       )
                      )
                      (call $18
                       (local.get $16)
                       (local.get $19)
                       (i32.const 1048880)
                      )
                      (unreachable)
                     )
                     (call $18
                      (local.get $35)
                      (local.get $33)
                      (i32.const 1048896)
                     )
                     (unreachable)
                    )
                    (call $18
                     (local.get $0)
                     (local.get $19)
                     (i32.const 1048864)
                    )
                    (unreachable)
                   )
                   (br_if $label10
                    (i32.gt_u
                     (local.get $27)
                     (local.get $30)
                    )
                   )
                  )
                 )
                )
                (i64.store align=4
                 (local.get $51)
                 (i64.load offset=4 align=4
                  (local.get $17)
                 )
                )
                (i32.store
                 (i32.add
                  (local.get $51)
                  (i32.const 8)
                 )
                 (i32.load
                  (i32.add
                   (local.get $17)
                   (i32.const 12)
                  )
                 )
                )
                (global.set $global$0
                 (i32.add
                  (local.get $17)
                  (i32.const 288)
                 )
                )
                (br $block34)
               )
               (call $18
                (local.get $20)
                (local.get $25)
                (i32.const 1049024)
               )
               (unreachable)
              )
              (call $29
               (local.get $29)
               (local.get $15)
               (i32.const 1048804)
              )
              (unreachable)
             )
             (local.set $19
              (i32.load offset=12
               (local.get $18)
              )
             )
             (if
              (local.tee $0
               (i32.load offset=16
                (local.get $18)
               )
              )
              (then
               (local.set $23
                (i32.add
                 (local.get $19)
                 (i32.shl
                  (local.get $0)
                  (i32.const 4)
                 )
                )
               )
               (local.set $0
                (i32.shl
                 (local.get $31)
                 (i32.const 3)
                )
               )
               (local.set $16
                (local.get $19)
               )
               (loop $label11
                (local.set $5
                 (f64.mul
                  (block $block37 (result f64)
                   (block $block35
                    (block $block38
                     (br $block37
                      (f64.add
                       (local.tee $11
                        (f64.add
                         (local.tee $9
                          (f64.mul
                           (local.tee $6
                            (f64.convert_i32_s
                             (i32.add
                              (i32.shr_u
                               (local.tee $20
                                (i32.add
                                 (block $block36 (result i32)
                                  (if
                                   (i64.ge_s
                                    (local.tee $69
                                     (i64.reinterpret_f64
                                      (local.tee $5
                                       (f64.add
                                        (f64.mul
                                         (local.tee $5
                                          (f64.load
                                           (local.get $16)
                                          )
                                         )
                                         (local.get $5)
                                        )
                                        (f64.mul
                                         (local.tee $5
                                          (f64.load offset=8
                                           (local.get $16)
                                          )
                                         )
                                         (local.get $5)
                                        )
                                       )
                                      )
                                     )
                                    )
                                    (i64.const 4503599627370496)
                                   )
                                   (then
                                    (br_if $block35
                                     (i64.gt_u
                                      (local.get $69)
                                      (i64.const 9218868437227405311)
                                     )
                                    )
                                    (local.set $15
                                     (i32.const -1023)
                                    )
                                    (if
                                     (i64.ne
                                      (local.tee $70
                                       (i64.shr_u
                                        (local.get $69)
                                        (i64.const 32)
                                       )
                                      )
                                      (i64.const 1072693248)
                                     )
                                     (then
                                      (br $block36
                                       (i32.wrap_i64
                                        (local.get $70)
                                       )
                                      )
                                     )
                                    )
                                    (drop
                                     (br_if $block36
                                      (i32.const 1072693248)
                                      (i32.wrap_i64
                                       (local.get $69)
                                      )
                                     )
                                    )
                                    (br $block37
                                     (f64.const 0)
                                    )
                                   )
                                  )
                                  (drop
                                   (br_if $block37
                                    (f64.div
                                     (f64.const -1)
                                     (f64.mul
                                      (local.get $5)
                                      (local.get $5)
                                     )
                                    )
                                    (f64.eq
                                     (local.get $5)
                                     (f64.const 0)
                                    )
                                   )
                                  )
                                  (br_if $block38
                                   (i64.lt_s
                                    (local.get $69)
                                    (i64.const 0)
                                   )
                                  )
                                  (local.set $15
                                   (i32.const -1077)
                                  )
                                  (i32.wrap_i64
                                   (i64.shr_u
                                    (local.tee $69
                                     (i64.reinterpret_f64
                                      (f64.mul
                                       (local.get $5)
                                       (f64.const 18014398509481984)
                                      )
                                     )
                                    )
                                    (i64.const 32)
                                   )
                                  )
                                 )
                                 (i32.const 614242)
                                )
                               )
                               (i32.const 20)
                              )
                              (local.get $15)
                             )
                            )
                           )
                           (f64.const 0.30102999566361177)
                          )
                         )
                         (local.tee $10
                          (f64.mul
                           (local.tee $8
                            (f64.reinterpret_i64
                             (i64.and
                              (i64.reinterpret_f64
                               (f64.sub
                                (local.tee $5
                                 (f64.add
                                  (f64.reinterpret_i64
                                   (i64.or
                                    (i64.and
                                     (local.get $69)
                                     (i64.const 4294967295)
                                    )
                                    (i64.shl
                                     (i64.extend_i32_u
                                      (i32.add
                                       (i32.and
                                        (local.get $20)
                                        (i32.const 1048575)
                                       )
                                       (i32.const 1072079006)
                                      )
                                     )
                                     (i64.const 32)
                                    )
                                   )
                                  )
                                  (f64.const -1)
                                 )
                                )
                                (local.tee $7
                                 (f64.mul
                                  (local.get $5)
                                  (f64.mul
                                   (local.get $5)
                                   (f64.const 0.5)
                                  )
                                 )
                                )
                               )
                              )
                              (i64.const -4294967296)
                             )
                            )
                           )
                           (f64.const 0.4342944818781689)
                          )
                         )
                        )
                       )
                       (f64.add
                        (f64.add
                         (local.get $10)
                         (f64.sub
                          (local.get $9)
                          (local.get $11)
                         )
                        )
                        (f64.add
                         (f64.mul
                          (local.tee $5
                           (f64.add
                            (f64.sub
                             (f64.sub
                              (local.get $5)
                              (local.get $8)
                             )
                             (local.get $7)
                            )
                            (f64.mul
                             (local.tee $5
                              (f64.div
                               (local.get $5)
                               (f64.add
                                (local.get $5)
                                (f64.const 2)
                               )
                              )
                             )
                             (f64.add
                              (local.get $7)
                              (f64.add
                               (f64.mul
                                (local.tee $5
                                 (f64.mul
                                  (local.tee $7
                                   (f64.mul
                                    (local.get $5)
                                    (local.get $5)
                                   )
                                  )
                                  (local.get $7)
                                 )
                                )
                                (f64.add
                                 (f64.mul
                                  (local.get $5)
                                  (f64.add
                                   (f64.mul
                                    (local.get $5)
                                    (f64.const 0.15313837699209373)
                                   )
                                   (f64.const 0.22222198432149784)
                                  )
                                 )
                                 (f64.const 0.3999999999940942)
                                )
                               )
                               (f64.mul
                                (local.get $7)
                                (f64.add
                                 (f64.mul
                                  (local.get $5)
                                  (f64.add
                                   (f64.mul
                                    (local.get $5)
                                    (f64.add
                                     (f64.mul
                                      (local.get $5)
                                      (f64.const 0.14798198605116586)
                                     )
                                     (f64.const 0.1818357216161805)
                                    )
                                   )
                                   (f64.const 0.2857142874366239)
                                  )
                                 )
                                 (f64.const 0.6666666666666735)
                                )
                               )
                              )
                             )
                            )
                           )
                          )
                          (f64.const 0.4342944818781689)
                         )
                         (f64.add
                          (f64.mul
                           (local.get $6)
                           (f64.const 3.694239077158931e-13)
                          )
                          (f64.mul
                           (f64.add
                            (local.get $5)
                            (local.get $8)
                           )
                           (f64.const 2.5082946711645275e-11)
                          )
                         )
                        )
                       )
                      )
                     )
                    )
                    (local.set $5
                     (f64.div
                      (f64.sub
                       (local.get $5)
                       (local.get $5)
                      )
                      (f64.const 0)
                     )
                    )
                   )
                   (local.get $5)
                  )
                  (f64.const 5)
                 )
                )
                (if
                 (i32.eq
                  (i32.load offset=24
                   (local.get $18)
                  )
                  (local.get $31)
                 )
                 (then
                  (local.set $32
                   (i32.const 0)
                  )
                  (global.set $global$0
                   (local.tee $15
                    (i32.sub
                     (global.get $global$0)
                     (i32.const 32)
                    )
                   )
                  )
                  (if
                   (i32.gt_u
                    (local.tee $21
                     (select
                      (local.tee $21
                       (i32.add
                        (local.tee $24
                         (i32.load
                          (local.tee $20
                           (i32.add
                            (local.get $18)
                            (i32.const 24)
                           )
                          )
                         )
                        )
                        (i32.const 1)
                       )
                      )
                      (local.tee $27
                       (i32.shl
                        (local.get $24)
                        (i32.const 1)
                       )
                      )
                      (i32.gt_u
                       (local.get $21)
                       (local.get $27)
                      )
                     )
                    )
                    (i32.const 536870911)
                   )
                   (then
                    (call $29
                     (i32.const 0)
                     (i32.const 0)
                     (i32.const 1049308)
                    )
                    (unreachable)
                   )
                  )
                  (block $block39
                   (call $29
                    (if (result i32)
                     (i32.le_u
                      (local.tee $27
                       (i32.shl
                        (local.tee $21
                         (select
                          (i32.const 4)
                          (local.get $21)
                          (i32.le_u
                           (local.get $21)
                           (i32.const 4)
                          )
                         )
                        )
                        (i32.const 3)
                       )
                      )
                      (i32.const 2147483640)
                     )
                     (then
                      (i32.store offset=24
                       (local.get $15)
                       (if (result i32)
                        (local.get $24)
                        (then
                         (i32.store offset=28
                          (local.get $15)
                          (i32.shl
                           (local.get $24)
                           (i32.const 3)
                          )
                         )
                         (i32.store offset=20
                          (local.get $15)
                          (i32.load offset=4
                           (local.get $20)
                          )
                         )
                         (i32.const 8)
                        )
                        (else
                         (i32.const 0)
                        )
                       )
                      )
                      (call $16
                       (i32.add
                        (local.get $15)
                        (i32.const 8)
                       )
                       (i32.const 8)
                       (local.get $27)
                       (i32.add
                        (local.get $15)
                        (i32.const 20)
                       )
                      )
                      (br_if $block39
                       (i32.ne
                        (i32.load offset=8
                         (local.get $15)
                        )
                        (i32.const 1)
                       )
                      )
                      (local.set $32
                       (i32.load offset=16
                        (local.get $15)
                       )
                      )
                      (i32.load offset=12
                       (local.get $15)
                      )
                     )
                     (else
                      (i32.const 0)
                     )
                    )
                    (local.get $32)
                    (i32.const 1049308)
                   )
                   (unreachable)
                  )
                  (local.set $24
                   (i32.load offset=12
                    (local.get $15)
                   )
                  )
                  (i32.store
                   (local.get $20)
                   (local.get $21)
                  )
                  (i32.store offset=4
                   (local.get $20)
                   (local.get $24)
                  )
                  (global.set $global$0
                   (i32.add
                    (local.get $15)
                    (i32.const 32)
                   )
                  )
                  (local.set $32
                   (i32.load offset=28
                    (local.get $18)
                   )
                  )
                 )
                )
                (f64.store
                 (i32.add
                  (local.get $0)
                  (local.get $32)
                 )
                 (local.get $5)
                )
                (i32.store offset=32
                 (local.get $18)
                 (local.tee $31
                  (i32.add
                   (local.get $31)
                   (i32.const 1)
                  )
                 )
                )
                (local.set $0
                 (i32.add
                  (local.get $0)
                  (i32.const 8)
                 )
                )
                (br_if $label11
                 (i32.ne
                  (local.tee $16
                   (i32.add
                    (local.get $16)
                    (i32.const 16)
                   )
                  )
                  (local.get $23)
                 )
                )
               )
              )
             )
             (if
              (local.tee $0
               (i32.load offset=8
                (local.get $18)
               )
              )
              (then
               (call $42
                (local.get $19)
                (i32.shl
                 (local.get $0)
                 (i32.const 4)
                )
               )
              )
             )
             (if
              (local.get $25)
              (then
               (call $42
                (local.get $38)
                (local.get $45)
               )
              )
             )
             (local.set $16
              (i32.add
               (select
                (local.get $4)
                (i32.const 0)
                (local.tee $44
                 (i32.sub
                  (local.get $44)
                  (i32.const 1)
                 )
                )
               )
               (local.get $37)
              )
             )
             (local.set $26
              (i32.add
               (local.get $26)
               (local.get $56)
              )
             )
             (local.set $19
              (local.get $37)
             )
             (br_if $label12
              (local.get $44)
             )
            )
            (br $block15)
           )
          )
          (i32.store offset=52
           (local.get $18)
           (i32.const 0)
          )
          (i32.store offset=40
           (local.get $18)
           (i32.const 1)
          )
          (i32.store offset=36
           (local.get $18)
           (i32.const 1049064)
          )
          (i64.store offset=44 align=4
           (local.get $18)
           (i64.const 4)
          )
          (call $26
           (i32.add
            (local.get $18)
            (i32.const 36)
           )
           (i32.const 1049072)
          )
          (unreachable)
         )
         (i32.store offset=40
          (local.get $18)
          (i32.const 3)
         )
         (i32.store offset=36
          (local.get $18)
          (i32.const 1049132)
         )
         (i64.store offset=48 align=4
          (local.get $18)
          (i64.const 2)
         )
         (i64.store offset=16
          (local.get $18)
          (i64.or
           (i64.extend_i32_u
            (local.get $18)
           )
           (i64.const 4294967296)
          )
         )
         (i64.store offset=8
          (local.get $18)
          (i64.or
           (i64.extend_i32_u
            (i32.add
             (local.get $18)
             (i32.const 4)
            )
           )
           (i64.const 4294967296)
          )
         )
         (i32.store offset=44
          (local.get $18)
          (i32.add
           (local.get $18)
           (i32.const 8)
          )
         )
         (call $26
          (i32.add
           (local.get $18)
           (i32.const 36)
          )
          (i32.const 1049156)
         )
         (unreachable)
        )
        (call $29
         (local.get $16)
         (local.get $0)
         (i32.const 1049172)
        )
        (unreachable)
       )
       (global.set $global$0
        (local.tee $0
         (i32.sub
          (global.get $global$0)
          (i32.const 48)
         )
        )
       )
       (i32.store offset=4
        (local.get $0)
        (local.get $15)
       )
       (i32.store
        (local.get $0)
        (local.get $19)
       )
       (i32.store offset=12
        (local.get $0)
        (i32.const 2)
       )
       (i32.store offset=8
        (local.get $0)
        (i32.const 1050604)
       )
       (br $block40)
      )
      (global.set $global$0
       (local.tee $0
        (i32.sub
         (global.get $global$0)
         (i32.const 48)
        )
       )
      )
      (i32.store offset=4
       (local.get $0)
       (local.get $2)
      )
      (i32.store
       (local.get $0)
       (local.get $15)
      )
      (i32.store offset=12
       (local.get $0)
       (i32.const 2)
      )
      (i32.store offset=8
       (local.get $0)
       (i32.const 1050552)
      )
      (br $block40)
     )
     (call $29
      (i32.const 8)
      (local.get $45)
      (i32.const 1049292)
     )
     (unreachable)
    )
    (i64.store align=4
     (local.get $50)
     (i64.load offset=24 align=4
      (local.get $18)
     )
    )
    (i32.store
     (i32.add
      (local.get $50)
      (i32.const 8)
     )
     (i32.load
      (i32.add
       (local.get $18)
       (i32.const 32)
      )
     )
    )
    (if
     (local.tee $0
      (i32.load offset=36
       (local.get $18)
      )
     )
     (then
      (call $42
       (i32.load offset=40
        (local.get $18)
       )
       (i32.shl
        (local.get $0)
        (i32.const 4)
       )
      )
     )
    )
    (if
     (local.tee $0
      (i32.load offset=48
       (local.get $18)
      )
     )
     (then
      (call $42
       (i32.load offset=52
        (local.get $18)
       )
       (i32.shl
        (local.get $0)
        (i32.const 4)
       )
      )
     )
    )
    (if
     (local.tee $0
      (i32.load offset=60
       (local.get $18)
      )
     )
     (then
      (call $42
       (i32.load offset=64
        (local.get $18)
       )
       (i32.shl
        (local.get $0)
        (i32.const 2)
       )
      )
     )
    )
    (global.set $global$0
     (i32.add
      (local.get $18)
      (i32.const 80)
     )
    )
    (br $block41)
   )
   (i64.store offset=20 align=4
    (local.get $0)
    (i64.const 2)
   )
   (i64.store offset=40
    (local.get $0)
    (i64.or
     (i64.extend_i32_u
      (i32.add
       (local.get $0)
       (i32.const 4)
      )
     )
     (i64.const 4294967296)
    )
   )
   (i64.store offset=32
    (local.get $0)
    (i64.or
     (i64.extend_i32_u
      (local.get $0)
     )
     (i64.const 4294967296)
    )
   )
   (i32.store offset=16
    (local.get $0)
    (i32.add
     (local.get $0)
     (i32.const 32)
    )
   )
   (call $26
    (i32.add
     (local.get $0)
     (i32.const 8)
    )
    (i32.const 1049188)
   )
   (unreachable)
  )
  (if
   (local.get $4)
   (then
    (call $42
     (local.get $3)
     (i32.shl
      (local.get $4)
      (i32.const 3)
     )
    )
   )
  )
  (if
   (local.get $2)
   (then
    (call $42
     (local.get $1)
     (i32.shl
      (local.get $2)
      (i32.const 3)
     )
    )
   )
  )
  (block $block42
   (if
    (i32.le_u
     (local.tee $1
      (i32.load offset=4
       (local.get $34)
      )
     )
     (local.tee $0
      (i32.load offset=12
       (local.get $34)
      )
     )
    )
    (then
     (local.set $1
      (i32.load offset=8
       (local.get $34)
      )
     )
     (br $block42)
    )
   )
   (local.set $2
    (i32.shl
     (local.get $1)
     (i32.const 3)
    )
   )
   (local.set $3
    (i32.load offset=8
     (local.get $34)
    )
   )
   (if
    (i32.eqz
     (local.get $0)
    )
    (then
     (local.set $1
      (i32.const 8)
     )
     (call $42
      (local.get $3)
      (local.get $2)
     )
     (br $block42)
    )
   )
   (br_if $block42
    (local.tee $1
     (call $31
      (local.get $3)
      (local.get $2)
      (i32.const 8)
      (local.tee $2
       (i32.shl
        (local.get $0)
        (i32.const 3)
       )
      )
     )
    )
   )
   (call $29
    (i32.const 8)
    (local.get $2)
    (i32.const 1049432)
   )
   (unreachable)
  )
  (i32.store offset=4
   (local.get $36)
   (local.get $0)
  )
  (i32.store
   (local.get $36)
   (local.get $1)
  )
  (global.set $global$0
   (i32.add
    (local.get $34)
    (i32.const 16)
   )
  )
  (tuple.make 2
   (i32.load
    (local.get $36)
   )
   (block (result i32)
    (local.set $scratch_77
     (i32.load offset=4
      (local.get $36)
     )
    )
    (global.set $global$0
     (i32.add
      (local.get $36)
      (i32.const 16)
     )
    )
    (local.get $scratch_77)
   )
  )
 )
 (func $26 (param $0 i32) (param $1 i32)
  (local $2 i32)
  (local $3 i32)
  (local $4 i64)
  (global.set $global$0
   (local.tee $2
    (i32.sub
     (global.get $global$0)
     (i32.const 16)
    )
   )
  )
  (i32.store16 offset=12
   (local.get $2)
   (i32.const 1)
  )
  (i32.store offset=8
   (local.get $2)
   (local.get $1)
  )
  (i32.store offset=4
   (local.get $2)
   (local.get $0)
  )
  (global.set $global$0
   (local.tee $1
    (i32.sub
     (global.get $global$0)
     (i32.const 16)
    )
   )
  )
  (local.set $4
   (i64.load align=4
    (local.tee $0
     (i32.add
      (local.get $2)
      (i32.const 4)
     )
    )
   )
  )
  (i32.store offset=12
   (local.get $1)
   (local.get $0)
  )
  (i64.store offset=4 align=4
   (local.get $1)
   (local.get $4)
  )
  (global.set $global$0
   (local.tee $0
    (i32.sub
     (global.get $global$0)
     (i32.const 16)
    )
   )
  )
  (local.set $3
   (i32.load offset=12
    (local.tee $2
     (i32.load
      (local.tee $1
       (i32.add
        (local.get $1)
        (i32.const 4)
       )
      )
     )
    )
   )
  )
  (block $block3
   (block $block2
    (block $block1
     (block $block
      (br_table $block $block1 $block2
       (i32.load offset=4
        (local.get $2)
       )
      )
     )
     (br_if $block2
      (local.get $3)
     )
     (local.set $2
      (i32.const 1)
     )
     (local.set $3
      (i32.const 0)
     )
     (br $block3)
    )
    (br_if $block2
     (local.get $3)
    )
    (local.set $3
     (i32.load offset=4
      (local.tee $2
       (i32.load
        (local.get $2)
       )
      )
     )
    )
    (local.set $2
     (i32.load
      (local.get $2)
     )
    )
    (br $block3)
   )
   (i32.store
    (local.get $0)
    (i32.const -2147483648)
   )
   (i32.store offset=12
    (local.get $0)
    (local.get $1)
   )
   (call $15
    (local.get $0)
    (i32.const 1050176)
    (i32.load offset=4
     (local.get $1)
    )
    (i32.load8_u offset=8
     (local.tee $0
      (i32.load offset=8
       (local.get $1)
      )
     )
    )
    (i32.load8_u offset=9
     (local.get $0)
    )
   )
   (unreachable)
  )
  (i32.store offset=4
   (local.get $0)
   (local.get $3)
  )
  (i32.store
   (local.get $0)
   (local.get $2)
  )
  (call $15
   (local.get $0)
   (i32.const 1050148)
   (i32.load offset=4
    (local.get $1)
   )
   (i32.load8_u offset=8
    (local.tee $0
     (i32.load offset=8
      (local.get $1)
     )
    )
   )
   (i32.load8_u offset=9
    (local.get $0)
   )
  )
  (unreachable)
 )
 (func $27 (param $0 i32)
  (local $1 i32)
  (if
   (i32.ne
    (i32.or
     (local.tee $1
      (i32.load
       (local.get $0)
      )
     )
     (i32.const -2147483648)
    )
    (i32.const -2147483648)
   )
   (then
    (call $42
     (i32.load offset=4
      (local.get $0)
     )
     (local.get $1)
    )
   )
  )
 )
 (func $28 (param $0 i32)
  (local $1 i32)
  (if
   (local.tee $1
    (i32.load
     (local.get $0)
    )
   )
   (then
    (call $42
     (i32.load offset=4
      (local.get $0)
     )
     (local.get $1)
    )
   )
  )
 )
 (func $29 (param $0 i32) (param $1 i32) (param $2 i32)
  (if
   (i32.eqz
    (local.get $0)
   )
   (then
    (global.set $global$0
     (local.tee $0
      (i32.sub
       (global.get $global$0)
       (i32.const 32)
      )
     )
    )
    (i32.store offset=24
     (local.get $0)
     (i32.const 0)
    )
    (i32.store offset=12
     (local.get $0)
     (i32.const 1)
    )
    (i32.store offset=8
     (local.get $0)
     (i32.const 1050224)
    )
    (i64.store offset=16 align=4
     (local.get $0)
     (i64.const 4)
    )
    (call $26
     (i32.add
      (local.get $0)
      (i32.const 8)
     )
     (local.get $2)
    )
    (unreachable)
   )
  )
  (call $46
   (local.get $0)
   (local.get $1)
  )
  (unreachable)
 )
 (func $30 (param $0 i32)
  (i32.store offset=16
   (local.get $0)
   (i32.const 0)
  )
  (i64.store offset=8 align=4
   (local.get $0)
   (i64.const 0)
  )
  (i64.store align=4
   (local.get $0)
   (i64.const 17179869184)
  )
 )
 (func $31 (param $0 i32) (param $1 i32) (param $2 i32) (param $3 i32) (result i32)
  (local $4 i32)
  (local $5 i32)
  (local $6 i32)
  (local $7 i32)
  (local $8 i32)
  (local $9 i32)
  (local $scratch i32)
  (block $block2 (result i32)
   (block $block5
    (block $block11
     (block $block13
      (block $block12
       (block $block
        (if
         (i32.ge_u
          (local.tee $4
           (i32.and
            (local.tee $6
             (i32.load
              (local.tee $5
               (i32.sub
                (local.get $0)
                (i32.const 4)
               )
              )
             )
            )
            (i32.const -8)
           )
          )
          (i32.add
           (select
            (i32.const 4)
            (i32.const 8)
            (local.tee $7
             (i32.and
              (local.get $6)
              (i32.const 3)
             )
            )
           )
           (local.get $1)
          )
         )
         (then
          (br_if $block
           (select
            (local.get $7)
            (i32.const 0)
            (i32.lt_u
             (local.tee $9
              (i32.add
               (local.get $1)
               (i32.const 39)
              )
             )
             (local.get $4)
            )
           )
          )
          (block $block3
           (block $block1
            (if
             (i32.ge_u
              (local.get $2)
              (i32.const 9)
             )
             (then
              (br_if $block1
               (local.tee $8
                (call $7
                 (local.get $2)
                 (local.get $3)
                )
               )
              )
              (br $block2
               (i32.const 0)
              )
             )
            )
            (br_if $block3
             (i32.gt_u
              (local.get $3)
              (i32.const -65588)
             )
            )
            (local.set $1
             (select
              (i32.const 16)
              (i32.and
               (i32.add
                (local.get $3)
                (i32.const 11)
               )
               (i32.const -8)
              )
              (i32.lt_u
               (local.get $3)
               (i32.const 11)
              )
             )
            )
            (block $block4
             (if
              (i32.eqz
               (local.get $7)
              )
              (then
               (br_if $block4
                (i32.or
                 (i32.or
                  (i32.lt_u
                   (local.get $1)
                   (i32.const 256)
                  )
                  (i32.lt_u
                   (local.get $4)
                   (i32.or
                    (local.get $1)
                    (i32.const 4)
                   )
                  )
                 )
                 (i32.ge_u
                  (i32.sub
                   (local.get $4)
                   (local.get $1)
                  )
                  (i32.const 131073)
                 )
                )
               )
               (br $block5)
              )
             )
             (local.set $7
              (i32.add
               (local.tee $2
                (i32.sub
                 (local.get $0)
                 (i32.const 8)
                )
               )
               (local.get $4)
              )
             )
             (block $block6
              (block $block9
               (block $block7
                (block $block8
                 (if
                  (i32.gt_u
                   (local.get $1)
                   (local.get $4)
                  )
                  (then
                   (br_if $block6
                    (i32.eq
                     (local.get $7)
                     (i32.load
                      (i32.const 1051456)
                     )
                    )
                   )
                   (br_if $block7
                    (i32.eq
                     (local.get $7)
                     (i32.load
                      (i32.const 1051452)
                     )
                    )
                   )
                   (br_if $block4
                    (i32.and
                     (local.tee $6
                      (i32.load offset=4
                       (local.get $7)
                      )
                     )
                     (i32.const 2)
                    )
                   )
                   (br_if $block4
                    (i32.lt_u
                     (local.tee $4
                      (i32.add
                       (local.tee $6
                        (i32.and
                         (local.get $6)
                         (i32.const -8)
                        )
                       )
                       (local.get $4)
                      )
                     )
                     (local.get $1)
                    )
                   )
                   (call $8
                    (local.get $7)
                    (local.get $6)
                   )
                   (br_if $block8
                    (i32.lt_u
                     (local.tee $3
                      (i32.sub
                       (local.get $4)
                       (local.get $1)
                      )
                     )
                     (i32.const 16)
                    )
                   )
                   (i32.store
                    (local.get $5)
                    (i32.or
                     (i32.or
                      (local.get $1)
                      (i32.and
                       (i32.load
                        (local.get $5)
                       )
                       (i32.const 1)
                      )
                     )
                     (i32.const 2)
                    )
                   )
                   (i32.store offset=4
                    (local.tee $1
                     (i32.add
                      (local.get $1)
                      (local.get $2)
                     )
                    )
                    (i32.or
                     (local.get $3)
                     (i32.const 3)
                    )
                   )
                   (i32.store offset=4
                    (local.tee $2
                     (i32.add
                      (local.get $2)
                      (local.get $4)
                     )
                    )
                    (i32.or
                     (i32.load offset=4
                      (local.get $2)
                     )
                     (i32.const 1)
                    )
                   )
                   (call $5
                    (local.get $1)
                    (local.get $3)
                   )
                   (br $block5)
                  )
                 )
                 (br_if $block9
                  (i32.gt_u
                   (local.tee $3
                    (i32.sub
                     (local.get $4)
                     (local.get $1)
                    )
                   )
                   (i32.const 15)
                  )
                 )
                 (br $block5)
                )
                (i32.store
                 (local.get $5)
                 (i32.or
                  (i32.or
                   (local.get $4)
                   (i32.and
                    (i32.load
                     (local.get $5)
                    )
                    (i32.const 1)
                   )
                  )
                  (i32.const 2)
                 )
                )
                (i32.store offset=4
                 (local.tee $1
                  (i32.add
                   (local.get $2)
                   (local.get $4)
                  )
                 )
                 (i32.or
                  (i32.load offset=4
                   (local.get $1)
                  )
                  (i32.const 1)
                 )
                )
                (br $block5)
               )
               (br_if $block4
                (i32.lt_u
                 (local.tee $4
                  (i32.add
                   (i32.load
                    (i32.const 1051444)
                   )
                   (local.get $4)
                  )
                 )
                 (local.get $1)
                )
               )
               (block $block10
                (if
                 (i32.le_u
                  (local.tee $3
                   (i32.sub
                    (local.get $4)
                    (local.get $1)
                   )
                  )
                  (i32.const 15)
                 )
                 (then
                  (i32.store
                   (local.get $5)
                   (i32.or
                    (i32.or
                     (i32.and
                      (local.get $6)
                      (i32.const 1)
                     )
                     (local.get $4)
                    )
                    (i32.const 2)
                   )
                  )
                  (i32.store offset=4
                   (local.tee $1
                    (i32.add
                     (local.get $2)
                     (local.get $4)
                    )
                   )
                   (i32.or
                    (i32.load offset=4
                     (local.get $1)
                    )
                    (i32.const 1)
                   )
                  )
                  (local.set $3
                   (i32.const 0)
                  )
                  (local.set $1
                   (i32.const 0)
                  )
                  (br $block10)
                 )
                )
                (i32.store
                 (local.get $5)
                 (i32.or
                  (i32.or
                   (local.get $1)
                   (i32.and
                    (local.get $6)
                    (i32.const 1)
                   )
                  )
                  (i32.const 2)
                 )
                )
                (i32.store offset=4
                 (local.tee $1
                  (i32.add
                   (local.get $1)
                   (local.get $2)
                  )
                 )
                 (i32.or
                  (local.get $3)
                  (i32.const 1)
                 )
                )
                (i32.store
                 (local.tee $2
                  (i32.add
                   (local.get $2)
                   (local.get $4)
                  )
                 )
                 (local.get $3)
                )
                (i32.store offset=4
                 (local.get $2)
                 (i32.and
                  (i32.load offset=4
                   (local.get $2)
                  )
                  (i32.const -2)
                 )
                )
               )
               (i32.store
                (i32.const 1051452)
                (local.get $1)
               )
               (i32.store
                (i32.const 1051444)
                (local.get $3)
               )
               (br $block5)
              )
              (i32.store
               (local.get $5)
               (i32.or
                (i32.or
                 (local.get $1)
                 (i32.and
                  (local.get $6)
                  (i32.const 1)
                 )
                )
                (i32.const 2)
               )
              )
              (i32.store offset=4
               (local.tee $1
                (i32.add
                 (local.get $1)
                 (local.get $2)
                )
               )
               (i32.or
                (local.get $3)
                (i32.const 3)
               )
              )
              (i32.store offset=4
               (local.get $7)
               (i32.or
                (i32.load offset=4
                 (local.get $7)
                )
                (i32.const 1)
               )
              )
              (call $5
               (local.get $1)
               (local.get $3)
              )
              (br $block5)
             )
             (br_if $block11
              (i32.gt_u
               (local.tee $4
                (i32.add
                 (i32.load
                  (i32.const 1051448)
                 )
                 (local.get $4)
                )
               )
               (local.get $1)
              )
             )
            )
            (br_if $block3
             (i32.eqz
              (local.tee $1
               (call $0
                (local.get $3)
               )
              )
             )
            )
            (br $block2
             (block (result i32)
              (local.set $scratch
               (call $2
                (local.get $1)
                (local.get $0)
                (select
                 (local.tee $1
                  (i32.add
                   (select
                    (i32.const -4)
                    (i32.const -8)
                    (i32.and
                     (local.tee $1
                      (i32.load
                       (local.get $5)
                      )
                     )
                     (i32.const 3)
                    )
                   )
                   (i32.and
                    (local.get $1)
                    (i32.const -8)
                   )
                  )
                 )
                 (local.get $3)
                 (i32.lt_u
                  (local.get $1)
                  (local.get $3)
                 )
                )
               )
              )
              (call $3
               (local.get $0)
              )
              (local.get $scratch)
             )
            )
           )
           (drop
            (call $2
             (local.get $8)
             (local.get $0)
             (select
              (local.get $3)
              (local.get $1)
              (i32.gt_u
               (local.get $1)
               (local.get $3)
              )
             )
            )
           )
           (br_if $block12
            (i32.lt_u
             (local.tee $3
              (i32.and
               (local.tee $2
                (i32.load
                 (local.get $5)
                )
               )
               (i32.const -8)
              )
             )
             (i32.add
              (local.get $1)
              (select
               (i32.const 4)
               (i32.const 8)
               (local.tee $2
                (i32.and
                 (local.get $2)
                 (i32.const 3)
                )
               )
              )
             )
            )
           )
           (br_if $block13
            (select
             (local.get $2)
             (i32.const 0)
             (i32.gt_u
              (local.get $3)
              (local.get $9)
             )
            )
           )
           (call $3
            (local.get $0)
           )
          )
          (br $block2
           (local.get $8)
          )
         )
        )
        (call $22
         (i32.const 1049897)
         (i32.const 46)
         (i32.const 1049944)
        )
        (unreachable)
       )
       (call $22
        (i32.const 1049960)
        (i32.const 46)
        (i32.const 1050008)
       )
       (unreachable)
      )
      (call $22
       (i32.const 1049897)
       (i32.const 46)
       (i32.const 1049944)
      )
      (unreachable)
     )
     (call $22
      (i32.const 1049960)
      (i32.const 46)
      (i32.const 1050008)
     )
     (unreachable)
    )
    (i32.store
     (local.get $5)
     (i32.or
      (i32.or
       (local.get $1)
       (i32.and
        (local.get $6)
        (i32.const 1)
       )
      )
      (i32.const 2)
     )
    )
    (i32.store offset=4
     (local.tee $2
      (i32.add
       (local.get $1)
       (local.get $2)
      )
     )
     (i32.or
      (local.tee $1
       (i32.sub
        (local.get $4)
        (local.get $1)
       )
      )
      (i32.const 1)
     )
    )
    (i32.store
     (i32.const 1051448)
     (local.get $1)
    )
    (i32.store
     (i32.const 1051456)
     (local.get $2)
    )
    (br $block2
     (local.get $0)
    )
   )
   (local.get $0)
  )
 )
 (func $32 (param $0 i32) (param $1 i32) (param $2 i32)
  (if
   (local.get $1)
   (then
    (call $42
     (local.get $0)
     (local.get $1)
    )
   )
  )
 )
 (func $33 (param $0 i32) (param $1 i32) (param $2 i32) (result i32)
  (call_indirect $0 (type $2)
   (i32.load offset=28
    (local.get $0)
   )
   (local.get $1)
   (local.get $2)
   (i32.load offset=12
    (i32.load offset=32
     (local.get $0)
    )
   )
  )
 )
 (func $34 (param $0 i32) (param $1 i32) (param $2 i32)
  (v128.store
   (local.get $0)
   (f64x2.add
    (v128.load
     (local.get $1)
    )
    (v128.load
     (local.get $2)
    )
   )
  )
 )
 (func $35 (param $0 i32) (param $1 i32) (param $2 i32)
  (v128.store
   (local.get $0)
   (f64x2.sub
    (v128.load
     (local.get $1)
    )
    (v128.load
     (local.get $2)
    )
   )
  )
 )
 (func $36 (param $0 i32) (param $1 f64) (param $2 f64)
  (v128.store
   (local.get $0)
   (f64x2.replace_lane 1
    (f64x2.splat
     (local.get $1)
    )
    (local.get $2)
   )
  )
 )
 (func $37 (param $0 i32) (param $1 i32) (result i32)
  (block $block (result i32)
   (if
    (i32.ge_u
     (local.get $1)
     (i32.const 9)
    )
    (then
     (br $block
      (call $7
       (local.get $1)
       (local.get $0)
      )
     )
    )
   )
   (call $0
    (local.get $0)
   )
  )
 )
 (func $38 (param $0 i32) (param $1 i32)
  (i64.store offset=8
   (local.get $0)
   (i64.const -3009991045896884125)
  )
  (i64.store
   (local.get $0)
   (i64.const 3999679090963090256)
  )
 )
 (func $39 (param $0 i32) (param $1 i32)
  (i64.store offset=8
   (local.get $0)
   (i64.const 7199936582794304877)
  )
  (i64.store
   (local.get $0)
   (i64.const -5076933981314334344)
  )
 )
 (func $40 (param $0 i32) (param $1 i32)
  (i32.store offset=4
   (local.get $0)
   (i32.const 1050132)
  )
  (i32.store
   (local.get $0)
   (local.get $1)
  )
 )
 (func $41 (param $0 i32) (param $1 i32) (result i32)
  (call $33
   (local.get $1)
   (i32.load
    (local.get $0)
   )
   (i32.load offset=4
    (local.get $0)
   )
  )
 )
 (func $42 (param $0 i32) (param $1 i32)
  (local $2 i32)
  (local $3 i32)
  (block $block1
   (block $block
    (if
     (i32.ge_u
      (local.tee $3
       (i32.and
        (local.tee $2
         (i32.load
          (i32.sub
           (local.get $0)
           (i32.const 4)
          )
         )
        )
        (i32.const -8)
       )
      )
      (i32.add
       (select
        (i32.const 4)
        (i32.const 8)
        (local.tee $2
         (i32.and
          (local.get $2)
          (i32.const 3)
         )
        )
       )
       (local.get $1)
      )
     )
     (then
      (br_if $block
       (select
        (local.get $2)
        (i32.const 0)
        (i32.gt_u
         (local.get $3)
         (i32.add
          (local.get $1)
          (i32.const 39)
         )
        )
       )
      )
      (call $3
       (local.get $0)
      )
      (br $block1)
     )
    )
    (call $22
     (i32.const 1049897)
     (i32.const 46)
     (i32.const 1049944)
    )
    (unreachable)
   )
   (call $22
    (i32.const 1049960)
    (i32.const 46)
    (i32.const 1050008)
   )
   (unreachable)
  )
 )
 (func $43 (param $0 i32) (param $1 i32) (result i32)
  (local $2 i32)
  (local $3 i32)
  (local $4 i32)
  (local $5 i32)
  (local $6 i32)
  (local $7 i32)
  (local $8 i32)
  (local $9 i32)
  (local $10 i32)
  (local $11 i32)
  (local $12 i32)
  (local $scratch i32)
  (local $scratch_14 i32)
  (local.set $6
   (i32.load
    (local.get $0)
   )
  )
  (local.set $2
   (local.get $1)
  )
  (global.set $global$0
   (local.tee $9
    (i32.sub
     (global.get $global$0)
     (i32.const 16)
    )
   )
  )
  (local.set $3
   (i32.const 10)
  )
  (if
   (i32.ge_u
    (local.tee $0
     (local.get $6)
    )
    (i32.const 1000)
   )
   (then
    (local.set $1
     (local.get $0)
    )
    (loop $label
     (i32.store8
      (i32.sub
       (local.tee $4
        (i32.add
         (i32.add
          (local.get $9)
          (i32.const 6)
         )
         (local.get $3)
        )
       )
       (i32.const 3)
      )
      (i32.load8_u
       (i32.add
        (local.tee $8
         (i32.shl
          (local.tee $5
           (i32.div_u
            (i32.and
             (local.tee $7
              (i32.sub
               (local.get $1)
               (i32.mul
                (local.tee $0
                 (i32.div_u
                  (local.get $1)
                  (i32.const 10000)
                 )
                )
                (i32.const 10000)
               )
              )
             )
             (i32.const 65535)
            )
            (i32.const 100)
           )
          )
          (i32.const 1)
         )
        )
        (i32.const 1050301)
       )
      )
     )
     (i32.store8
      (i32.sub
       (local.get $4)
       (i32.const 4)
      )
      (i32.load8_u
       (i32.add
        (local.get $8)
        (i32.const 1050300)
       )
      )
     )
     (i32.store8
      (i32.sub
       (local.get $4)
       (i32.const 1)
      )
      (i32.load8_u
       (i32.add
        (local.tee $7
         (i32.shl
          (i32.and
           (i32.sub
            (local.get $7)
            (i32.mul
             (local.get $5)
             (i32.const 100)
            )
           )
           (i32.const 65535)
          )
          (i32.const 1)
         )
        )
        (i32.const 1050301)
       )
      )
     )
     (i32.store8
      (i32.sub
       (local.get $4)
       (i32.const 2)
      )
      (i32.load8_u
       (i32.add
        (local.get $7)
        (i32.const 1050300)
       )
      )
     )
     (local.set $3
      (i32.sub
       (local.get $3)
       (i32.const 4)
      )
     )
     (br_if $label
      (block (result i32)
       (local.set $scratch
        (i32.gt_u
         (local.get $1)
         (i32.const 9999999)
        )
       )
       (local.set $1
        (local.get $0)
       )
       (local.get $scratch)
      )
     )
    )
   )
  )
  (block $block
   (if
    (i32.le_u
     (local.get $0)
     (i32.const 9)
    )
    (then
     (local.set $1
      (local.get $0)
     )
     (br $block)
    )
   )
   (i32.store8
    (i32.add
     (i32.add
      (local.get $3)
      (local.get $9)
     )
     (i32.const 5)
    )
    (i32.load8_u
     (i32.add
      (local.tee $0
       (i32.shl
        (i32.and
         (i32.sub
          (local.get $0)
          (i32.mul
           (local.tee $1
            (i32.div_u
             (i32.and
              (local.get $0)
              (i32.const 65535)
             )
             (i32.const 100)
            )
           )
           (i32.const 100)
          )
         )
         (i32.const 65535)
        )
        (i32.const 1)
       )
      )
      (i32.const 1050301)
     )
    )
   )
   (i32.store8
    (i32.add
     (local.tee $3
      (i32.sub
       (local.get $3)
       (i32.const 2)
      )
     )
     (i32.add
      (local.get $9)
      (i32.const 6)
     )
    )
    (i32.load8_u
     (i32.add
      (local.get $0)
      (i32.const 1050300)
     )
    )
   )
  )
  (if
   (i32.eqz
    (select
     (i32.const 0)
     (local.get $6)
     (local.get $1)
    )
   )
   (then
    (i32.store8
     (i32.add
      (local.tee $3
       (i32.sub
        (local.get $3)
        (i32.const 1)
       )
      )
      (i32.add
       (local.get $9)
       (i32.const 6)
      )
     )
     (i32.load8_u
      (i32.add
       (i32.and
        (i32.shl
         (local.get $1)
         (i32.const 1)
        )
        (i32.const 30)
       )
       (i32.const 1050301)
      )
     )
    )
   )
  )
  (local.set $scratch_14
   (block $block1 (result i32)
    (local.set $4
     (i32.add
      (i32.add
       (local.get $9)
       (i32.const 6)
      )
      (local.get $3)
     )
    )
    (local.set $6
     (i32.sub
      (i32.const 10)
      (local.get $3)
     )
    )
    (local.set $3
     (select
      (i32.const 43)
      (i32.const 1114112)
      (local.tee $1
       (i32.and
        (local.tee $0
         (i32.load offset=20
          (local.get $2)
         )
        )
        (i32.const 1)
       )
      )
     )
    )
    (local.set $7
     (i32.eqz
      (i32.eqz
       (i32.and
        (local.get $0)
        (i32.const 4)
       )
      )
     )
    )
    (if
     (i32.eqz
      (i32.load
       (local.get $2)
      )
     )
     (then
      (drop
       (br_if $block1
        (i32.const 1)
        (call $23
         (local.tee $0
          (i32.load offset=28
           (local.get $2)
          )
         )
         (local.tee $1
          (i32.load offset=32
           (local.get $2)
          )
         )
         (local.get $3)
         (local.get $7)
        )
       )
      )
      (br $block1
       (call_indirect $0 (type $2)
        (local.get $0)
        (local.get $4)
        (local.get $6)
        (i32.load offset=12
         (local.get $1)
        )
       )
      )
     )
    )
    (block $block4
     (block $block3
      (block $block2
       (if
        (i32.le_u
         (local.tee $5
          (i32.load offset=4
           (local.get $2)
          )
         )
         (local.tee $8
          (i32.add
           (local.get $1)
           (local.get $6)
          )
         )
        )
        (then
         (br_if $block2
          (i32.eqz
           (call $23
            (local.tee $0
             (i32.load offset=28
              (local.get $2)
             )
            )
            (local.tee $1
             (i32.load offset=32
              (local.get $2)
             )
            )
            (local.get $3)
            (local.get $7)
           )
          )
         )
         (br $block1
          (i32.const 1)
         )
        )
       )
       (br_if $block3
        (i32.eqz
         (i32.and
          (local.get $0)
          (i32.const 8)
         )
        )
       )
       (local.set $11
        (i32.load offset=16
         (local.get $2)
        )
       )
       (i32.store offset=16
        (local.get $2)
        (i32.const 48)
       )
       (local.set $12
        (i32.load8_u offset=24
         (local.get $2)
        )
       )
       (local.set $1
        (i32.const 1)
       )
       (i32.store8 offset=24
        (local.get $2)
        (i32.const 1)
       )
       (br_if $block4
        (call $23
         (local.tee $0
          (i32.load offset=28
           (local.get $2)
          )
         )
         (local.tee $10
          (i32.load offset=32
           (local.get $2)
          )
         )
         (local.get $3)
         (local.get $7)
        )
       )
       (local.set $1
        (i32.add
         (i32.sub
          (local.get $5)
          (local.get $8)
         )
         (i32.const 1)
        )
       )
       (block $block5
        (loop $label1
         (br_if $block5
          (i32.eqz
           (local.tee $1
            (i32.sub
             (local.get $1)
             (i32.const 1)
            )
           )
          )
         )
         (br_if $label1
          (i32.eqz
           (call_indirect $0 (type $1)
            (local.get $0)
            (i32.const 48)
            (i32.load offset=16
             (local.get $10)
            )
           )
          )
         )
        )
        (br $block1
         (i32.const 1)
        )
       )
       (drop
        (br_if $block1
         (i32.const 1)
         (call_indirect $0 (type $2)
          (local.get $0)
          (local.get $4)
          (local.get $6)
          (i32.load offset=12
           (local.get $10)
          )
         )
        )
       )
       (i32.store8 offset=24
        (local.get $2)
        (local.get $12)
       )
       (i32.store offset=16
        (local.get $2)
        (local.get $11)
       )
       (br $block1
        (i32.const 0)
       )
      )
      (local.set $1
       (call_indirect $0 (type $2)
        (local.get $0)
        (local.get $4)
        (local.get $6)
        (i32.load offset=12
         (local.get $1)
        )
       )
      )
      (br $block4)
     )
     (local.set $0
      (i32.sub
       (local.get $5)
       (local.get $8)
      )
     )
     (block $block8
      (block $block7
       (block $block6
        (br_table $block6 $block7 $block8
         (i32.sub
          (local.tee $1
           (select
            (i32.const 1)
            (local.tee $1
             (i32.load8_u offset=24
              (local.get $2)
             )
            )
            (i32.eq
             (local.get $1)
             (i32.const 3)
            )
           )
          )
          (i32.const 1)
         )
        )
       )
       (local.set $1
        (local.get $0)
       )
       (local.set $0
        (i32.const 0)
       )
       (br $block8)
      )
      (local.set $1
       (i32.shr_u
        (local.get $0)
        (i32.const 1)
       )
      )
      (local.set $0
       (i32.shr_u
        (i32.add
         (local.get $0)
         (i32.const 1)
        )
        (i32.const 1)
       )
      )
     )
     (local.set $1
      (i32.add
       (local.get $1)
       (i32.const 1)
      )
     )
     (local.set $8
      (i32.load offset=16
       (local.get $2)
      )
     )
     (local.set $5
      (i32.load offset=32
       (local.get $2)
      )
     )
     (local.set $2
      (i32.load offset=28
       (local.get $2)
      )
     )
     (block $block9
      (loop $label2
       (br_if $block9
        (i32.eqz
         (local.tee $1
          (i32.sub
           (local.get $1)
           (i32.const 1)
          )
         )
        )
       )
       (br_if $label2
        (i32.eqz
         (call_indirect $0 (type $1)
          (local.get $2)
          (local.get $8)
          (i32.load offset=16
           (local.get $5)
          )
         )
        )
       )
      )
      (br $block1
       (i32.const 1)
      )
     )
     (local.set $1
      (i32.const 1)
     )
     (br_if $block4
      (call $23
       (local.get $2)
       (local.get $5)
       (local.get $3)
       (local.get $7)
      )
     )
     (br_if $block4
      (call_indirect $0 (type $2)
       (local.get $2)
       (local.get $4)
       (local.get $6)
       (i32.load offset=12
        (local.get $5)
       )
      )
     )
     (local.set $1
      (i32.const 0)
     )
     (loop $label3
      (drop
       (br_if $block1
        (i32.const 0)
        (i32.eq
         (local.get $0)
         (local.get $1)
        )
       )
      )
      (local.set $1
       (i32.add
        (local.get $1)
        (i32.const 1)
       )
      )
      (br_if $label3
       (i32.eqz
        (call_indirect $0 (type $1)
         (local.get $2)
         (local.get $8)
         (i32.load offset=16
          (local.get $5)
         )
        )
       )
      )
     )
     (br $block1
      (i32.lt_u
       (i32.sub
        (local.get $1)
        (i32.const 1)
       )
       (local.get $0)
      )
     )
    )
    (local.get $1)
   )
  )
  (global.set $global$0
   (i32.add
    (local.get $9)
    (i32.const 16)
   )
  )
  (local.get $scratch_14)
 )
 (func $44 (param $0 i32) (param $1 i32) (result i32)
  (call $4
   (local.get $0)
   (i32.const 1049832)
   (local.get $1)
  )
 )
 (func $45 (param $0 i32) (param $1 i32)
  (i64.store
   (local.get $0)
   (i64.load align=4
    (local.get $1)
   )
  )
 )
 (func $46 (param $0 i32) (param $1 i32)
  (call_indirect $0 (type $0)
   (local.get $0)
   (local.get $1)
   (select
    (local.tee $0
     (i32.load
      (i32.const 1051008)
     )
    )
    (i32.const 3)
    (local.get $0)
   )
  )
  (unreachable)
 )
 (func $47 (param $0 i32) (param $1 i32)
  (i32.store
   (local.get $0)
   (i32.const 0)
  )
 )
 (func $48 (param $0 f64) (result f64)
  (local $1 f64)
  (local $2 f64)
  (local $3 f64)
  (local $4 f64)
  (local $5 f64)
  (local $6 i32)
  (local $7 i32)
  (global.set $global$0
   (local.tee $6
    (i32.sub
     (global.get $global$0)
     (i32.const 32)
    )
   )
  )
  (block $block4
   (if
    (i32.ge_u
     (local.tee $7
      (i32.and
       (i32.wrap_i64
        (i64.shr_u
         (i64.reinterpret_f64
          (local.get $0)
         )
         (i64.const 32)
        )
       )
       (i32.const 2147483647)
      )
     )
     (i32.const 1072243196)
    )
    (then
     (block $block1
      (block $block
       (block $block3
        (block $block2
         (if
          (i32.le_u
           (local.get $7)
           (i32.const 2146435071)
          )
          (then
           (call $1
            (i32.add
             (local.get $6)
             (i32.const 8)
            )
            (local.get $0)
           )
           (local.set $2
            (f64.load offset=24
             (local.get $6)
            )
           )
           (local.set $3
            (f64.mul
             (local.tee $0
              (f64.mul
               (local.tee $1
                (f64.load offset=8
                 (local.get $6)
                )
               )
               (local.get $1)
              )
             )
             (local.get $0)
            )
           )
           (br_table $block $block1 $block2 $block3
            (i32.sub
             (i32.and
              (i32.load offset=16
               (local.get $6)
              )
              (i32.const 3)
             )
             (i32.const 1)
            )
           )
          )
         )
         (local.set $0
          (f64.sub
           (local.get $0)
           (local.get $0)
          )
         )
         (br $block4)
        )
        (local.set $0
         (f64.neg
          (f64.add
           (local.tee $5
            (f64.sub
             (f64.const 1)
             (local.tee $4
              (f64.mul
               (local.get $0)
               (f64.const 0.5)
              )
             )
            )
           )
           (f64.add
            (f64.sub
             (f64.sub
              (f64.const 1)
              (local.get $5)
             )
             (local.get $4)
            )
            (f64.sub
             (f64.mul
              (local.get $0)
              (f64.add
               (f64.mul
                (local.get $0)
                (f64.add
                 (f64.mul
                  (local.get $0)
                  (f64.add
                   (f64.mul
                    (local.get $0)
                    (f64.const 2.480158728947673e-05)
                   )
                   (f64.const -0.001388888888887411)
                  )
                 )
                 (f64.const 0.0416666666666666)
                )
               )
               (f64.mul
                (f64.mul
                 (local.get $3)
                 (local.get $3)
                )
                (f64.add
                 (f64.mul
                  (local.get $0)
                  (f64.add
                   (f64.mul
                    (local.get $0)
                    (f64.const -1.1359647557788195e-11)
                   )
                   (f64.const 2.087572321298175e-09)
                  )
                 )
                 (f64.const -2.7557314351390663e-07)
                )
               )
              )
             )
             (f64.mul
              (local.get $1)
              (local.get $2)
             )
            )
           )
          )
         )
        )
        (br $block4)
       )
       (local.set $0
        (f64.sub
         (local.get $1)
         (f64.add
          (f64.mul
           (local.tee $1
            (f64.mul
             (local.get $1)
             (local.get $0)
            )
           )
           (f64.const 0.16666666666666632)
          )
          (f64.sub
           (f64.mul
            (local.get $0)
            (f64.sub
             (f64.mul
              (local.get $2)
              (f64.const 0.5)
             )
             (f64.mul
              (local.get $1)
              (f64.add
               (f64.mul
                (f64.mul
                 (local.get $0)
                 (local.get $3)
                )
                (f64.add
                 (f64.mul
                  (local.get $0)
                  (f64.const 1.58969099521155e-10)
                 )
                 (f64.const -2.5050760253406863e-08)
                )
               )
               (f64.add
                (f64.mul
                 (local.get $0)
                 (f64.add
                  (f64.mul
                   (local.get $0)
                   (f64.const 2.7557313707070068e-06)
                  )
                  (f64.const -1.984126982985795e-04)
                 )
                )
                (f64.const 0.00833333333332249)
               )
              )
             )
            )
           )
           (local.get $2)
          )
         )
        )
       )
       (br $block4)
      )
      (local.set $0
       (f64.add
        (local.tee $5
         (f64.sub
          (f64.const 1)
          (local.tee $4
           (f64.mul
            (local.get $0)
            (f64.const 0.5)
           )
          )
         )
        )
        (f64.add
         (f64.sub
          (f64.sub
           (f64.const 1)
           (local.get $5)
          )
          (local.get $4)
         )
         (f64.sub
          (f64.mul
           (local.get $0)
           (f64.add
            (f64.mul
             (local.get $0)
             (f64.add
              (f64.mul
               (local.get $0)
               (f64.add
                (f64.mul
                 (local.get $0)
                 (f64.const 2.480158728947673e-05)
                )
                (f64.const -0.001388888888887411)
               )
              )
              (f64.const 0.0416666666666666)
             )
            )
            (f64.mul
             (f64.mul
              (local.get $3)
              (local.get $3)
             )
             (f64.add
              (f64.mul
               (local.get $0)
               (f64.add
                (f64.mul
                 (local.get $0)
                 (f64.const -1.1359647557788195e-11)
                )
                (f64.const 2.087572321298175e-09)
               )
              )
              (f64.const -2.7557314351390663e-07)
             )
            )
           )
          )
          (f64.mul
           (local.get $1)
           (local.get $2)
          )
         )
        )
       )
      )
      (br $block4)
     )
     (local.set $0
      (f64.neg
       (f64.sub
        (local.get $1)
        (f64.add
         (f64.mul
          (local.tee $1
           (f64.mul
            (local.get $1)
            (local.get $0)
           )
          )
          (f64.const 0.16666666666666632)
         )
         (f64.sub
          (f64.mul
           (local.get $0)
           (f64.sub
            (f64.mul
             (local.get $2)
             (f64.const 0.5)
            )
            (f64.mul
             (local.get $1)
             (f64.add
              (f64.mul
               (f64.mul
                (local.get $0)
                (local.get $3)
               )
               (f64.add
                (f64.mul
                 (local.get $0)
                 (f64.const 1.58969099521155e-10)
                )
                (f64.const -2.5050760253406863e-08)
               )
              )
              (f64.add
               (f64.mul
                (local.get $0)
                (f64.add
                 (f64.mul
                  (local.get $0)
                  (f64.const 2.7557313707070068e-06)
                 )
                 (f64.const -1.984126982985795e-04)
                )
               )
               (f64.const 0.00833333333332249)
              )
             )
            )
           )
          )
          (local.get $2)
         )
        )
       )
      )
     )
     (br $block4)
    )
   )
   (if
    (i32.ge_u
     (local.get $7)
     (i32.const 1045430272)
    )
    (then
     (local.set $0
      (f64.add
       (f64.mul
        (f64.mul
         (local.tee $1
          (f64.mul
           (local.get $0)
           (local.get $0)
          )
         )
         (local.get $0)
        )
        (f64.add
         (f64.mul
          (local.get $1)
          (f64.add
           (f64.mul
            (f64.mul
             (local.get $1)
             (f64.mul
              (local.get $1)
              (local.get $1)
             )
            )
            (f64.add
             (f64.mul
              (local.get $1)
              (f64.const 1.58969099521155e-10)
             )
             (f64.const -2.5050760253406863e-08)
            )
           )
           (f64.add
            (f64.mul
             (local.get $1)
             (f64.add
              (f64.mul
               (local.get $1)
               (f64.const 2.7557313707070068e-06)
              )
              (f64.const -1.984126982985795e-04)
             )
            )
            (f64.const 0.00833333333332249)
           )
          )
         )
         (f64.const -0.16666666666666632)
        )
       )
       (local.get $0)
      )
     )
     (br $block4)
    )
   )
   (if
    (i32.ge_u
     (local.get $7)
     (i32.const 1048576)
    )
    (then
     (f64.store offset=8
      (local.get $6)
      (f64.add
       (local.get $0)
       (f64.const 1329227995784915872903807e12)
      )
     )
     (drop
      (f64.load offset=8
       (local.get $6)
      )
     )
     (br $block4)
    )
   )
   (f64.store offset=8
    (local.get $6)
    (f64.mul
     (local.get $0)
     (f64.const 7.52316384526264e-37)
    )
   )
   (drop
    (f64.load offset=8
     (local.get $6)
    )
   )
  )
  (global.set $global$0
   (i32.add
    (local.get $6)
    (i32.const 32)
   )
  )
  (local.get $0)
 )
 (func $49 (param $0 f64) (result f64)
  (local $1 f64)
  (local $2 f64)
  (local $3 f64)
  (local $4 f64)
  (local $5 i32)
  (local $6 i32)
  (local $7 i32)
  (local $scratch f64)
  (global.set $global$0
   (local.tee $5
    (i32.sub
     (global.get $global$0)
     (i32.const 32)
    )
   )
  )
  (local.set $scratch
   (block $block6 (result f64)
    (block $block2
     (block $block1
      (block $block4
       (block $block3
        (block $block
         (if
          (i32.ge_u
           (local.tee $6
            (i32.and
             (i32.wrap_i64
              (i64.shr_u
               (i64.reinterpret_f64
                (local.get $0)
               )
               (i64.const 32)
              )
             )
             (i32.const 2147483647)
            )
           )
           (i32.const 1072243196)
          )
          (then
           (br_if $block
            (i32.gt_u
             (local.get $6)
             (i32.const 2146435071)
            )
           )
           (call $1
            (i32.add
             (local.get $5)
             (i32.const 8)
            )
            (local.get $0)
           )
           (local.set $2
            (f64.load offset=24
             (local.get $5)
            )
           )
           (local.set $0
            (f64.mul
             (local.tee $1
              (f64.load offset=8
               (local.get $5)
              )
             )
             (local.get $1)
            )
           )
           (br_table $block1 $block2 $block3 $block4
            (i32.sub
             (i32.and
              (i32.load offset=16
               (local.get $5)
              )
              (i32.const 3)
             )
             (i32.const 1)
            )
           )
          )
         )
         (local.set $7
          (f64.ge
           (local.get $0)
           (f64.const -2147483648)
          )
         )
         (if
          (i32.eqz
           (select
            (select
             (i32.const 2147483647)
             (select
              (block $block5 (result i32)
               (if
                (f64.lt
                 (f64.abs
                  (local.get $0)
                 )
                 (f64.const 2147483648)
                )
                (then
                 (br $block5
                  (i32.trunc_f64_s
                   (local.get $0)
                  )
                 )
                )
               )
               (i32.const -2147483648)
              )
              (i32.const -2147483648)
              (local.get $7)
             )
             (f64.gt
              (local.get $0)
              (f64.const 2147483647)
             )
            )
            (i32.const 0)
            (f64.eq
             (local.get $0)
             (local.get $0)
            )
           )
          )
          (then
           (drop
            (br_if $block6
             (f64.const 1)
             (i32.lt_u
              (local.get $6)
              (i32.const 1044816030)
             )
            )
           )
          )
         )
         (br $block6
          (f64.add
           (local.tee $3
            (f64.sub
             (f64.const 1)
             (local.tee $2
              (f64.mul
               (local.tee $1
                (f64.mul
                 (local.get $0)
                 (local.get $0)
                )
               )
               (f64.const 0.5)
              )
             )
            )
           )
           (f64.add
            (f64.sub
             (f64.sub
              (f64.const 1)
              (local.get $3)
             )
             (local.get $2)
            )
            (f64.add
             (f64.mul
              (local.get $1)
              (f64.add
               (f64.mul
                (local.get $1)
                (f64.add
                 (f64.mul
                  (local.get $1)
                  (f64.add
                   (f64.mul
                    (local.get $1)
                    (f64.const 2.480158728947673e-05)
                   )
                   (f64.const -0.001388888888887411)
                  )
                 )
                 (f64.const 0.0416666666666666)
                )
               )
               (f64.mul
                (f64.mul
                 (local.tee $2
                  (f64.mul
                   (local.get $1)
                   (local.get $1)
                  )
                 )
                 (local.get $2)
                )
                (f64.add
                 (f64.mul
                  (local.get $1)
                  (f64.add
                   (f64.mul
                    (local.get $1)
                    (f64.const -1.1359647557788195e-11)
                   )
                   (f64.const 2.087572321298175e-09)
                  )
                 )
                 (f64.const -2.7557314351390663e-07)
                )
               )
              )
             )
             (f64.mul
              (local.get $0)
              (f64.const -0)
             )
            )
           )
          )
         )
        )
        (br $block6
         (f64.sub
          (local.get $0)
          (local.get $0)
         )
        )
       )
       (br $block6
        (f64.sub
         (local.get $1)
         (f64.add
          (f64.mul
           (local.tee $1
            (f64.mul
             (local.get $1)
             (local.get $0)
            )
           )
           (f64.const 0.16666666666666632)
          )
          (f64.sub
           (f64.mul
            (local.get $0)
            (f64.sub
             (f64.mul
              (local.get $2)
              (f64.const 0.5)
             )
             (f64.mul
              (local.get $1)
              (f64.add
               (f64.mul
                (f64.mul
                 (local.get $0)
                 (f64.mul
                  (local.get $0)
                  (local.get $0)
                 )
                )
                (f64.add
                 (f64.mul
                  (local.get $0)
                  (f64.const 1.58969099521155e-10)
                 )
                 (f64.const -2.5050760253406863e-08)
                )
               )
               (f64.add
                (f64.mul
                 (local.get $0)
                 (f64.add
                  (f64.mul
                   (local.get $0)
                   (f64.const 2.7557313707070068e-06)
                  )
                  (f64.const -1.984126982985795e-04)
                 )
                )
                (f64.const 0.00833333333332249)
               )
              )
             )
            )
           )
           (local.get $2)
          )
         )
        )
       )
      )
      (br $block6
       (f64.add
        (local.tee $4
         (f64.sub
          (f64.const 1)
          (local.tee $3
           (f64.mul
            (local.get $0)
            (f64.const 0.5)
           )
          )
         )
        )
        (f64.add
         (f64.sub
          (f64.sub
           (f64.const 1)
           (local.get $4)
          )
          (local.get $3)
         )
         (f64.sub
          (f64.mul
           (local.get $0)
           (f64.add
            (f64.mul
             (local.get $0)
             (f64.add
              (f64.mul
               (local.get $0)
               (f64.add
                (f64.mul
                 (local.get $0)
                 (f64.const 2.480158728947673e-05)
                )
                (f64.const -0.001388888888887411)
               )
              )
              (f64.const 0.0416666666666666)
             )
            )
            (f64.mul
             (f64.mul
              (local.tee $3
               (f64.mul
                (local.get $0)
                (local.get $0)
               )
              )
              (local.get $3)
             )
             (f64.add
              (f64.mul
               (local.get $0)
               (f64.add
                (f64.mul
                 (local.get $0)
                 (f64.const -1.1359647557788195e-11)
                )
                (f64.const 2.087572321298175e-09)
               )
              )
              (f64.const -2.7557314351390663e-07)
             )
            )
           )
          )
          (f64.mul
           (local.get $1)
           (local.get $2)
          )
         )
        )
       )
      )
     )
     (br $block6
      (f64.neg
       (f64.sub
        (local.get $1)
        (f64.add
         (f64.mul
          (local.tee $1
           (f64.mul
            (local.get $1)
            (local.get $0)
           )
          )
          (f64.const 0.16666666666666632)
         )
         (f64.sub
          (f64.mul
           (local.get $0)
           (f64.sub
            (f64.mul
             (local.get $2)
             (f64.const 0.5)
            )
            (f64.mul
             (local.get $1)
             (f64.add
              (f64.mul
               (f64.mul
                (local.get $0)
                (f64.mul
                 (local.get $0)
                 (local.get $0)
                )
               )
               (f64.add
                (f64.mul
                 (local.get $0)
                 (f64.const 1.58969099521155e-10)
                )
                (f64.const -2.5050760253406863e-08)
               )
              )
              (f64.add
               (f64.mul
                (local.get $0)
                (f64.add
                 (f64.mul
                  (local.get $0)
                  (f64.const 2.7557313707070068e-06)
                 )
                 (f64.const -1.984126982985795e-04)
                )
               )
               (f64.const 0.00833333333332249)
              )
             )
            )
           )
          )
          (local.get $2)
         )
        )
       )
      )
     )
    )
    (f64.neg
     (f64.add
      (local.tee $4
       (f64.sub
        (f64.const 1)
        (local.tee $3
         (f64.mul
          (local.get $0)
          (f64.const 0.5)
         )
        )
       )
      )
      (f64.add
       (f64.sub
        (f64.sub
         (f64.const 1)
         (local.get $4)
        )
        (local.get $3)
       )
       (f64.sub
        (f64.mul
         (local.get $0)
         (f64.add
          (f64.mul
           (local.get $0)
           (f64.add
            (f64.mul
             (local.get $0)
             (f64.add
              (f64.mul
               (local.get $0)
               (f64.const 2.480158728947673e-05)
              )
              (f64.const -0.001388888888887411)
             )
            )
            (f64.const 0.0416666666666666)
           )
          )
          (f64.mul
           (f64.mul
            (local.tee $3
             (f64.mul
              (local.get $0)
              (local.get $0)
             )
            )
            (local.get $3)
           )
           (f64.add
            (f64.mul
             (local.get $0)
             (f64.add
              (f64.mul
               (local.get $0)
               (f64.const -1.1359647557788195e-11)
              )
              (f64.const 2.087572321298175e-09)
             )
            )
            (f64.const -2.7557314351390663e-07)
           )
          )
         )
        )
        (f64.mul
         (local.get $1)
         (local.get $2)
        )
       )
      )
     )
    )
   )
  )
  (global.set $global$0
   (i32.add
    (local.get $5)
    (i32.const 32)
   )
  )
  (local.get $scratch)
 )
 ;; custom section "producers", size 114
 ;; features section: mutable-globals, simd, sign-ext, reference-types, multivalue
)

