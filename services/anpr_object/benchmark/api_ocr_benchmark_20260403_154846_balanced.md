# OCR Text Extraction API Benchmark Report

**Date:** 2026-04-03T15:48:46.233184

**API URL:** http://localhost:9003

**Dataset:** /home/harsha/abhishek/vlm-video-captioning/ANPR/anpr-inference-service/Indian vehicle license plate dataset/google_images

**OCR Plate Text Mode:** balanced

## Summary

- **Concurrency Level:** 3 concurrent user(s)
- **Total Images Tested:** 276
- **Successful:** 238 (86.2%)
- **Total Time:** 45.69s
- **Throughput:** 6.04 images/sec

## Accuracy Metrics

- **Average Accuracy:** 79.57%
- **Perfect Match (100%):** 134 (56.3%)
- **High Accuracy (>80%):** 175 (73.5%)
- **Valid Indian Plates:** 238 (100.0%)

## Performance Metrics

- **Avg Response Time:** 0.511s
- **Min Response Time:** 0.189s
- **Max Response Time:** 7.721s
- **Effective Throughput:** 5.21 successful/sec

## Detailed Results

| Image | Ground Truth | Predicted | Accuracy | Valid | Time (s) |
|-------|--------------|-----------|----------|-------|----------|
| 0073797c-a755-4972-b76b-8ef2b31d44ab___new_IMG_20160315_071740.jpg.jpeg | KA19TR02 | KA19TR220102011 | 46.7% | ✓ | 7.704 |
| 00b42b2c-f193-4863-b92c-0245cbc816da___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano-Petrol-Review-Images-Black-Front-Angle.jpg | TN21AT0492 | TN21AT0492 | 100.0% | ✓ | 7.642 |
| 03273806-bb1e-48da-8c8b-a0133a90197a___2014-Skoda-Yeti-Test-Drive.jpg.jpeg | MH20CS9817 | MH20CS9817 | 100.0% | ✓ | 7.721 |
| 0369b20e-b432-4409-90f9-2420877aa386___8151536c79159a1557421da5f27f9f0e.jpg.jpeg | KL05AK3300 | KLO5AK3300 | 90.0% | ✓ | 0.374 |
| 07064c2c-2aa3-4419-91a4-92916de8e54c___mahindra-scorpio-old-car-500x500.jpg.jpeg | MP09CP9052 | G06P360P | 0.0% | ✓ | 0.290 |
| 07aaab79-71ee-4ea3-a9e6-640191183947___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_1208484d1392541449-nissan-terrano-official-review-img_20140215_181708.jpg | KA18P2987 | 2987KA18P | 11.1% | ✓ | 0.485 |
| 07f6d77a-652e-4885-8520-6d405d2f712f___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_840539077_1_1080x720_nissan-terrano-xl-d-thp-110-ps-2016-diesel-.jpg | UP50AS4535 | UP50AS453 | 90.0% | ✓ | 0.406 |
| 0850c175-0b8d-47f2-801c-e29f1dbdb367___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_571.jpg | RJ27TC0530 | RJ27TC053 | 90.0% | ✓ | 0.246 |
| 092c497e-623d-499d-822a-4b01c657389b___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_nissan-terrano-rear-left-rim.jpg | KL63C8800 | C | 0.0% | ✓ | 0.328 |
| 0a0d1748-48cd-4114-90cb-b5baf0b3cbe4___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_147274518_15141875973_large.jpg | MH46X9996 | MH46X9996WO | 81.8% | ✓ | 0.520 |
| 0a720df9-e4ef-4e44-8c13-b39b9be8444d___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano-6.jpg | TN21AT0480 | TN21AT0480 | 100.0% | ✓ | 0.401 |
| 0b24d6ed-d32f-420d-bc18-03deece29073___2015-Maruti-Ciaz-Test-Drive-Review.jpg.jpeg | HR26CK8571 | HR26CK8571 | 100.0% | ✓ | 0.324 |
| 0b77e2a4-34db-486c-8d87-46186f5b7b88___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_nissan-terrano-diesel-review-motorzest-04.jpg | TN21AT0479 | N21AT0479 | 0.0% | ✓ | 0.484 |
| 0c9ebe94-827d-4c74-9950-6816e70d1bab___IMG_8883.jpg.jpeg | MH20BY3665 | MH20BY3665 | 100.0% | ✓ | 0.483 |
| 11186c4c-5f17-49b1-af37-a138a1e53794___12837944763_e7c6c6381a_z.jpg.jpeg | MH20CS1941 | MH20CS1941 | 100.0% | ✓ | 0.448 |
| 11832b13-d514-4f1d-967c-d00e76d21e9b___Yellow-Number-Plate-With-Black-Lettering.jpg.jpeg | TN59AQ1515 | N59AQ1515 | 0.0% | ✓ | 0.483 |
| 125b5cda-9691-47aa-8e57-f6d8a8537d83___fancy_number_plates_in_india.jpg.jpeg | TN66U8215 | TN66U8215 | 100.0% | ✓ | 0.364 |
| 12e26859-89e0-4c9b-9c59-ee793337464f___20.jpg.jpeg | HR26BR9044 | HR26BR9044 | 100.0% | ✓ | 0.288 |
| 1438b735-2d21-4879-8ed4-e5fa086e70c8___Maruti-Suzuki-Swift-Dzire-1.jpg.jpeg | HR26BP3543 | HR26BP3543 | 100.0% | ✓ | 0.340 |
| 18d2b870-7817-46da-a59a-6406c1b472c9___1033.png | KA51MJ8156 | KA51MJ8156 | 100.0% | ✓ | 0.388 |
| 1a0da3dc-6667-4092-bc2a-a541a06f1e90___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano9.jpg | KA04MN3622 | KAO4MN3622 | 90.0% | ✓ | 0.523 |
| 2153dfcb-5968-4bfc-86ba-7ee8ec2d25b1___163231d1425035435-skoda-rapid-tdi-remapped-code-6-tuning-ind-front.jpg.jpeg | TN07BV5200 | TNO7BV5200 | 90.0% | ✓ | 0.446 |
| 2302b12c-3d74-4d6c-b376-ba8ab308a8c3___big_215451_1320738450.jpg.jpeg | MH20BY4465 | MH20BY4465 | 100.0% | ✓ | 0.372 |
| 23d775eb-842f-4f57-ad7a-affafa21660a___215453-new-2016-maruti-ertiga-granite-grey-zdi-shvs-highway-champion-img_8168.jpg.jpeg | TN37CR4019 | TN37CR4019 | 100.0% | ✓ | 0.347 |
| 2424211e-e6ba-4478-8ed1-6a04e25499c7___main-qimg-7673aeac86d0ef987deb837552c5dd4c-1.png | KA03AB3380 | KA03AB3380 | 100.0% | ✓ | 0.507 |
| 2430703d-0fb3-4eb2-9765-4f9301f232cd___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Terrano-Duster-Rear-Comparison - Copy.jpg | TN19H3322 | 2222H6LN | 11.1% | ✓ | 0.362 |
| 266d82c7-1b64-44cd-b77c-c08d98175390___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_5c752daa8d168dec865fe0462bf2c0bb_large.jpg | MH21V9926 | MH21V9926 | 100.0% | ✓ | 0.315 |
| 26cfb3dd-3731-4fd0-a6c2-c44e3ce7498a___used_cars_in_keralaused_bikes_in_keralaautomotive_news7.jpg.jpeg | KL20K7561 | KL20K7561 | 100.0% | ✓ | 0.397 |
| 26d2b905-4055-41fe-a933-47db0e8a0a7f___37cba8c3f681f0cb5694f16a4c54aa7b.jpg.jpeg | HR26CF5868 | 1R26CF5868 | 90.0% | ✓ | 0.298 |
| 27a88c24-0bf2-48b0-98a8-fa396e951045___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_7T4G0491-e1376988169395-1024x577.jpg | TN21TC611 | TN21CTC61 | 55.6% | ✓ | 0.356 |
| 27bfd7b7-6e0c-454c-b9f3-7a44d36edb9c___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_nissan-terrano-rear-right-side-angle-view.jpg | DL6CM6683 | DL6CN6683 | 88.9% | ✓ | 0.513 |
| 28fc10e3-681b-4086-9c15-28934ae86f7e___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_nissan-terrano-amt_827x510_71478504527.jpg | DL14CTC0153 | DL14CTC015 | 90.9% | ✓ | 0.347 |
| 2c9306ab-3454-4ca0-89fd-db3f51dabcef___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano-front-three-quarters-on-plain-ground.jpg | RJ27TC0530 | RJ27TC0530 | 100.0% | ✓ | 0.338 |
| 2fbe224b-e775-4c74-b975-fd66d2397ef6___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano.jpg | RJ27TC0530 | R27TC530 | 10.0% | ✓ | 0.422 |
| 31847e78-6696-4dc2-be08-cb37d512c576___2010_Maruti_WagonR_Review.jpg.jpeg | MH06AW8929 | MHO6A8929 | 40.0% | ✓ | 0.381 |
| 3bc3f147-2d6b-484a-8481-4a49fd8afe23___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_844581764_1_1080x720_nissan-terrano-xv-d-thp-110-ps-2013-diesel-faridabad.jpg | HR51AY9099 | HR51AY909 | 90.0% | ✓ | 0.481 |
| 3d7cd880-ea4b-4d63-9706-bea1df54af20___135712d1400874352-skoda-rapid-1-6tdi-cr-mt-elegance-ultima-candy-white-white-monster-1911216_676392609084007_1524722860_o.jpg.jpeg | DL3CAY9324 | DL3CAY99324 | 63.6% | ✓ | 0.470 |
| 3f677265-d5fa-439f-be23-feaacce22c49___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_848126147_2_1080x720_nissan-terrano-xl-d-plus-2014-diesel-upload-photos.jpg | GJ05JH2501 | GJ05JH2507 | 90.0% | ✓ | 0.428 |
| 3fd93f90-f778-4f14-8b0f-67dd0be88a06___561644d1308248087-beware-fake-registration-numbers-imported-cars-proof-pg-3-257491_10150223034216705_685851704_7325894_6 - Copy.jpeg | WB06F9209 | WB06F9209 | 100.0% | ✓ | 0.444 |
| 41de2ad0-3785-488e-9f1f-75b76f446cc7___Maruti-Alto-K10-Photos-2.jpg.jpeg | HR26CT4063 | HR26CT4063 | 100.0% | ✓ | 0.337 |
| 422aaf6f-dd4a-450f-ac6b-65966ab512d0___1431956d1446017445-number-plates-merchandise-canvas-ink-gurgaon-edit-closed-thumb_img_4418_1024.jpg.jpeg | MH03BS7778 | MHD3BS7778 | 90.0% | ✓ | 0.456 |
| 49bdf0d9-4e64-41eb-9c19-eabdc4afb051___Maruti-Suzuki-Ciaz-Photos-30.JPG.jpeg | HR26CH3604 | HR26CH3604 | 100.0% | ✓ | 0.364 |
| 4eb236a3-6547-4103-b46f-3756d21128a9___06-Sanjay-Dutt.jpg.jpeg | MH02CB4545 | H02CB45 | 0.0% | ✓ | 0.369 |
| 4ed4509a-18ed-44cd-bfe3-b1cce982e365___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano-3.jpg | TN21AT0480 | TN21AT0480KAKENNISSANTOL149514955E | 29.4% | ✓ | 0.440 |
| 51d58875-0bdc-4488-84d9-8b7ad5883863___new_Maruti-Suzuki-Ignis-Rear-view-94486.jpg.jpeg | HR26DA0471 | HR26DA0471 | 100.0% | ✓ | 0.352 |
| 5231116f-7a07-4ce2-b437-d34b497e7969___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_844329566_3_1080x720_nissan-terrano-xe-d-2014-diesel-.jpg | HR11F7575 | HR11F7575 | 100.0% | ✓ | 0.376 |
| 52abe2c4-fde3-45b2-beec-f070e6ed98c7___Volkswagen-Vento-Rear-view-46765.jpg.jpeg | MH14EP4660 | MH14EP4660 | 100.0% | ✓ | 0.435 |
| 52b45060-9645-47c7-882d-b69a5bf07eff___1006372d1351343820-my-new-maruti-swift-zxi-21102012241.jpg.jpeg | KA031351 | KA031351 | 100.0% | ✓ | 0.385 |
| 5345a45f-8d1b-4984-b5ee-a69fc4cbda6a___new__bd7f7862-d727-11e7-ad30-e18a56154311.jpg.jpeg | MH15TC554 | MH15TC554 | 100.0% | ✓ | 0.384 |
| 55aa4ee8-13fd-45bd-b83b-651ac665d71a___200560387_4_1000x700_2012-toyota-innova-diesel-130000-kms-cars.jpg.jpeg | KL16J3636 | KL16J33636 | 60.0% | ✓ | 0.358 |
| 56b46df3-223c-42c8-a6bf-e95e5bac29cc___9214a754343030395508e104d7948167_large.jpg.jpeg | PB08CX2959 | OPB08CX2959 | 0.0% | ✓ | 0.432 |
| 57e92915-935a-4a91-91f8-f3ad70040a6e___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_847071203_1_1080x720_nissan-terrano-xv-d-thp-premium-110-ps-2015-diesel-ghaziabad.jpg | UP16CT2233 | UP16CT2233 | 100.0% | ✓ | 0.376 |
| 581e2224-3640-4bad-9d90-c2076a9f64f4___617609.jpg.jpeg | KA03MX5058 | 6000HOMGKA03MX5058 | 5.6% | ✓ | 0.499 |
| 59a40238-6b01-4bb9-9faf-3b595e67bcb3___3d-font-number-plate-500x500.jpg.jpeg | MH13BN4348 | MH1334348BN | 36.4% | ✓ | 0.468 |
| 59e121dc-ebc6-4e4c-ba88-7fe04b569218___Autopsyches-Skoda-Laura-6.jpg.jpeg | DL3CAY2231 | DL3CAY2231NO | 83.3% | ✓ | 0.320 |
| 5ac9f7b9-f7cc-4161-bbd1-7f377bb890bf___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano-John-Abraham.jpg | MH03CB6467 | MH03CB6467 | 100.0% | ✓ | 0.372 |
| 5b49ae32-c799-45e3-b139-9c56acf6370d___2014-Volkswagen-Polo-Review.jpg.jpeg | MH14EH7958 | MH14EH7958 | 100.0% | ✓ | 0.381 |
| 5c85a40f-ce81-44b4-82b2-dfdac281217d___new_Skoda-Octavia-1.8-TSI-5.jpg.jpeg | MH20CS1941 | MH20CS1941 | 100.0% | ✓ | 0.473 |
| 60842fa7-0a2e-4407-b41c-933d93bdbb35___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_nissan-terrano-red-rear.jpg | RJ27TC0530 | RJ27TC0530 | 100.0% | ✓ | 0.379 |
| 6578ae52-8431-4854-ba24-756888891756___1496414894_2017-maruti-suzuki-ciaz.jpg.jpeg | HR26TC7099 | HR26TC709 | 90.0% | ✓ | 0.519 |
| 658af519-5290-4ae5-8dd6-694bab9aa458___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_848765390_1_1080x720_nissan-terrano-xl-d-plus-2014-diesel-coimbatore.jpg | TN59BE0939 | RRCARS | 0.0% | ✓ | 0.610 |
| 6767eddd-bf2f-4270-aa1f-6146738895ab___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano-20300.jpg | RJ27TC0530 | RJ27TC0530 | 100.0% | ✓ | 0.371 |
| 684f079f-6085-420c-8e4e-e1aed270fc05___Toyota-Fortuner-Number-Plates-Design.jpg.jpeg | KL53E964 | KL53E964 | 100.0% | ✓ | 0.434 |
| 69b56ab9-accd-4c67-aa7a-a884c018e5ab___speedex-number-plates-nandanam-chennai-car-number-plate-dealers-3zh7n2s.jpg.jpeg | KL54H369 | KL541H369 | 44.4% | ✓ | 0.457 |
| 6bf2730d-82bf-43ac-8802-b61315d85b96___2017-Skoda-Octavia-rear-high-revealed-for-India-images.jpg.jpeg | MH20EE7598 | MH20EE7598 | 100.0% | ✓ | 0.434 |
| 6ec9d264-9eab-4027-bdcc-6f71c842ee75___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_846635891_1_1080x720_nissan-terrano-xl-d-thp-110-ps-2013-diesel-dehra-dun.jpg | UK07BA7252 | UKO7BA7252IND | 69.2% | ✓ | 0.440 |
| 6ed8b283-5cf5-4727-9969-c79f70f47e80___2-cars-same-numbers.jpg.jpeg | RJ27TA1143 | RJ271A111 | 70.0% | ✓ | 0.319 |
| 7073bd18-c0bf-47da-9e43-8c9d27831f11___new_Skoda-Laura-rear.jpg.jpeg | MH01AR5274 | MH011AR5274 | 36.4% | ✓ | 0.435 |
| 72894c2b-5999-4d87-baa4-e3507548e011___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_847602356_1_1080x720_nissan-terrano-xl-d-2014-diesel-bilaspur.jpg | CG10S0650 | CG10S0650 | 100.0% | ✓ | 0.497 |
| 760ba77a-8992-4574-8cba-6c081d9e3e19___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_04f4c639c871fba0b4cc6475604e40b7_large.jpg | RJ02CC0784 | RJ02CC0784 | 100.0% | ✓ | 0.388 |
| 7616cb19-df24-4d29-b679-5252391a547f___Skoda-Laura-rear.jpg.jpeg | MH01AR5274 | MHO1AR5274 | 90.0% | ✓ | 0.394 |
| 77d1f81a-bee6-487c-aff2-0efa31a9925c____bd7f7862-d727-11e7-ad30-e18a56154311.jpg.jpeg | MH15TC554 | MH15TC554 | 100.0% | ✓ | 0.423 |
| 7f768bfc-4c86-4fff-932d-c9871a82dce5___1431955d1446017445-number-plates-merchandise-canvas-ink-gurgaon-edit-closed-thumb_img_4422_1024.jpg.jpeg | MH03BS7778 | MHD3BS7778 | 90.0% | ✓ | 0.504 |
| 85d7e25d-5aa1-4e5f-9c4e-4e671f14ddbe___77cca4b6f3864b3928824a030ba101d2.jpg.jpeg | KL60N5344 | KL60N5344 | 100.0% | ✓ | 0.455 |
| 8625b831-13a1-4b7d-9466-826cd6d36c67___1304650d1414861991-number-plates-merchandise-canvas-ink-gurgaon-edit-closed-10471269_1537317779842156_5063668061078329632_n.jpg.jpeg | DL7CN5617 | DL7CN5617 | 100.0% | ✓ | 0.391 |
| 88294212-f11c-46a7-a91a-04c9da19efa6___DSC_8221-L.jpg.jpeg | WB74X7605 | WB74X7605 | 100.0% | ✓ | 0.426 |
| 8906af57-0d29-417d-ac46-408425527b57___VW-Vento-TDI-CNG-Spy-Pic-2.jpg.jpeg | MH14TCF300 | MH14TCF300 | 100.0% | ✓ | 0.388 |
| 89d26be7-258a-4470-b0fe-d1df57b20d55___new_Skoda-Octavia-5.jpg.jpeg | MH20CS1938 | MH20CS1938 | 100.0% | ✓ | 0.325 |
| 8b5431f9-7d6f-43ed-853d-350ca058ff2a___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_2014-Nissan-Terrano1.jpg | TN19TC94 | TN19TC94 | 100.0% | ✓ | 0.567 |
| 8c4371ee-a997-4b21-82c4-cb5e767c4d06___kia-sorento-spied-in-India.jpg.jpeg | AP02BP2454 | AP02BP2454 | 100.0% | ✓ | 0.343 |
| 8dbb3ea7-8512-49e4-9ede-d4647e3b7dd2___158712081_5_1000x700_toyota-innova-v-2010-7seat-life-tax-kerala_rev001.jpg.jpeg | KL02AF6363 | KL02ND | 40.0% | ✓ | 0.329 |
| 8edca6de-d964-4027-823d-f70d3f872412___12042943_1003224833031653_9073848483429898813_n.jpg.jpeg | KL55R2473 | KL55R2473 | 100.0% | ✓ | 0.608 |
| 967ba4a0-eb8e-4734-aa09-c7143d0404b6___39720161008_150234.jpg.jpeg | MH02BM5048 | MH02BM5048AUTOVIOWNEDCARIO | 38.5% | ✓ | 0.480 |
| 9cad4b47-6c7b-4e30-ab55-b7fa040beb56___135717d1400912281-skoda-rapid-1-6tdi-cr-mt-elegance-ultima-candy-white-white-monster-_mg_2993comp.jpg.jpeg | TN74AH1413 | TN74AH1413 | 100.0% | ✓ | 0.466 |
| 9ea92991-8fdf-4b46-aea1-bf08ecff247c___new_1107433d1373284570-my-modded-skoda-rapid-now-166-bhp-351-nm-119-whp-per-tonne-hdr_00005_normal.jpg.jpeg | TN07BU5427 | TN07BU5427 | 100.0% | ✓ | 0.381 |
| a1928183-9ebe-40f0-9a84-500aecc61474___2014-Skoda-Superb-facelift-spied-in-India.jpg.jpeg | MH20TC189B | MK20TC189B | 90.0% | ✓ | 0.280 |
| a285cdf6-4640-41ee-b3f3-b721f334466f___Skoda-Octavia-RS-launched-in-India-3.jpg.jpeg | MH20EE7598 | MH20EE7598 | 100.0% | ✓ | 0.371 |
| a2d49f53-4121-4627-b5d9-069f8b2f8490___840256171_1_1080x720_maruti-suzuki-wagon-r-vxi-with-abs-minor-2014-petrol-erode.jpg.jpeg | TN38BY4191 | TN38BY4191 | 100.0% | ✓ | 0.466 |
| a508ecb5-bfdd-46ba-8640-bb827b7c089c___Volkswagen-Polo.jpg.jpeg | MH14EH5819 | MH14EH5819 | 100.0% | ✓ | 0.397 |
| a615016b-4d07-46c7-a52b-8d4319fe3a10___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_nissan-terrano-rear-view.jpg | DL6CM6683 | 836C6 | 22.2% | ✓ | 0.426 |
| ab573806-1da7-4ccf-93c8-6082e0cc15c3___1107433d1373284570-my-modded-skoda-rapid-now-166-bhp-351-nm-119-whp-per-tonne-hdr_00005_normal.jpg.jpeg | TN07BU5427 | TN07BU5427 | 100.0% | ✓ | 0.422 |
| ad1f5cbc-7545-407e-8974-a03ddb7b7779___833542696_1_1080x720_mahindra-scorpio-vlx-4wd-airbag-bs-iv-2012-diesel-.jpg.jpeg | TN28BA9999 | TN28BA9999 | 100.0% | ✓ | 0.478 |
| af36405d-be93-4c8c-b1b2-5a0832535eeb___main-qimg-4caf07f4dfd9e2a2d279ddd54fb91986-c.jpg.jpeg | TN59AQ1515 | N59AQ1515 | 0.0% | ✓ | 0.475 |
| af608b5e-07d6-4bcf-84d0-f0cba9e54e12___Skoda-Superb-Rear-Profile.jpg.jpeg | MH20CS4946 | MH20CS4946 | 100.0% | ✓ | 0.472 |
| aff1f53b-3259-44f0-9718-05fb64dc8ba1___Untitled.png | KA03NA8385 | KA03NA8385 | 100.0% | ✓ | 0.405 |
| b117ba53-d563-49c2-806b-343666616815___new_Maruti-Suzuki-recalls-75419-Baleno-cars-1961-DZire-cars-only-AGS-variant-Motown-India-Bureau-2-810.jpg.jpeg | HR26CR3302 | HR26CR3302 | 100.0% | ✓ | 0.387 |
| b266759c-f9e9-47c1-9042-4c7aef7d441d___1711769d1514801394t-toyota-innova-crysta-2-4-gx-ownership-review-edit-10-000-km-service-done-vs.jpg.jpeg | KA21M5519 | KA21M5519 | 100.0% | ✓ | 0.392 |
| b5b41a4b-8913-4af6-9419-ac6398d266b0___161893107_2_1000x700_innava-with-fancy-number-upload-photos.jpg.jpeg | KL454455 | KL4554455 | 66.7% | ✓ | 0.418 |
| b79f870d-15c7-48e3-8874-7c0e53e29f75___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_754eab882b8290b29a0454dd45580a39_555X416_1 - Copy (2).jpg | HR26CP3135 | HR26CP313 | 90.0% | ✓ | 0.383 |
| b7fd8aaa-8621-449f-b99c-f45778f1a25c___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_1208485d1392541449-nissan-terrano-official-review-img_20140215_181507.jpg | KA18P2987 | KA18P2987 | 100.0% | ✓ | 0.365 |
| b9467810-4022-4cf5-87e7-fed0d63c8e2f___new_1658240d1500548920-close-look-2017-skoda-octavia-facelift-hands-free-parking-front-bumper.jpg.jpeg | MH20EE7597 | MH20EE7597 | 100.0% | ✓ | 0.503 |
| bb210ab4-795c-4b2b-9b6e-4597cbc67375___White-2BPlate.jpeg | MH01AV8866 | MHO1AV899 | 60.0% | ✓ | 0.437 |
| bb767d81-73b4-4c47-b619-7efde490b199___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_oem-name0.jpg | RJ27TC0530 | RJ27TC0530 | 100.0% | ✓ | 0.449 |
| bd742e42-bacc-430e-b9aa-9df666f5d9b3___kicker-647_010517112446.jpg.jpeg | AP28N3107 | AP28N3 | 66.7% | ✓ | 0.394 |
| bdad5f66-70ad-4241-9a62-bba1123c3486___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_1447945d1449508546t-my-nissan-terrano-85-ps-_img_000000_000000.jpg | KA05MQ92 | 6A05MO92 | 75.0% | ✓ | 0.388 |
| c03a7106-1fb9-471f-bbb0-2e8f4a29759a___38e7b1109eb976088e1df2f14ac4c08c.jpg - Copy.jpeg | KL13AA6340 | KL13AA634D | 90.0% | ✓ | 0.394 |
| c03a7106-1fb9-471f-bbb0-2e8f4a29759a___38e7b1109eb976088e1df2f14ac4c08c.jpg.jpeg | KL57A111 | KL57 | 50.0% | ✓ | 0.234 |
| c06d4d42-591f-429e-92b4-20e701f89c14___2014-VW-Polo-facelift-spied-India-rear.jpg.jpeg | MH14TCF460 | MH14TCF460 | 100.0% | ✓ | 0.293 |
| c2feb2b7-fbb3-4e1e-ad48-cc8112a1ab3f___Maruti_R3_On_Test.jpg.jpeg | HR99HA4575 | HR99HACTEMP4575 | 40.0% | ✓ | 0.371 |
| c30a1535-9d8f-465e-a7f4-aa726f19f0b1___new-maruti-alto-k10-2_625x300_41414071192.jpg.jpeg | HR26CT6702 | HR26CT6702 | 100.0% | ✓ | 0.439 |
| c41652d9-19ae-4a7d-8423-230340ac30bc___Skoda_Laura_DSG_Long_Term_Review-125.jpg.jpeg | MH20BN3525 | MHI2OBN3525 | 18.2% | ✓ | 0.340 |
| c795950c-766a-4a2e-9636-62d663d6f06b___62bf1edb36141f114521ec4bb4175579.jpg.jpeg | GJ7BB766 | G1788788 | 37.5% | ✓ | 0.424 |
| c8a79e91-8c48-42d0-9b80-137fd033493c___automakers-number-plates-nitin-gadkari_827x510_61522653557.jpg.jpeg | MH14TC947 | MH14TC94 | 88.9% | ✓ | 0.370 |
| car-wbs-AP02BP2454_00000.jpeg | AP02BP2454 | AP02BP2454 | 100.0% | ✓ | 0.456 |
| car-wbs-CH01AN0001_00000.png | CH01AN0001 | CH01AN0001 | 100.0% | ✓ | 0.485 |
| car-wbs-DL3CAY2231_00000.jpeg | DL3CAY2231 | DL3CAY2231NO | 83.3% | ✓ | 0.322 |
| car-wbs-DL3CAY9324_00000.jpeg | DL3CAY9324 | DL3CAY99324 | 63.6% | ✓ | 0.302 |
| car-wbs-DL6CM6683_00000.jpg | DL6CM6683 | DL6CN6683 | 88.9% | ✓ | 0.571 |
| car-wbs-DL6CM6683_00001.jpg | DL6CM6683 | 836C6 | 22.2% | ✓ | 0.504 |
| car-wbs-DL7CN5617_00000.png | DL7CN5617 | N | 0.0% | ✓ | 0.410 |
| car-wbs-DL8CX4850_00000.png | DL8CX4850 | DL8CX4850 | 100.0% | ✓ | 0.556 |
| car-wbs-GJ03JL0126_00000.png | GJ03JL0126 | GJ03JL0126 | 100.0% | ✓ | 0.533 |
| car-wbs-GJ05JH2501_00000.jpg | GJ05JH2501 | GJ05JH2507 | 90.0% | ✓ | 0.475 |
| car-wbs-GJ05JH2501_00001.jpg | GJ05JH2501 | GJ05JH250 | 90.0% | ✓ | 0.526 |
| car-wbs-HR11F7575_00000.jpg | HR11F7575 | HR11F7575 | 100.0% | ✓ | 0.479 |
| car-wbs-HR26AZ5927_00000.png | HR26AZ5927 | HR26AZ5927 | 100.0% | ✓ | 0.504 |
| car-wbs-HR26BC5514_00000.png | HR26BC5514 | HR26BC5514 | 100.0% | ✓ | 0.421 |
| car-wbs-HR26BC5514_00001.png | HR26BC5514 | HR26BC5514 | 100.0% | ✓ | 0.415 |
| car-wbs-HR26BP3543_00001.jpeg | HR26BP3543 | HR26BP3543 | 100.0% | ✓ | 0.430 |
| car-wbs-HR26BR9044_00000.png | HR26BR9044 | HR26BR9044 | 100.0% | ✓ | 0.451 |
| car-wbs-HR26CB1900_00000.png | HR26CB1900 | HR | 20.0% | ✓ | 0.452 |
| car-wbs-HR26CE1485_00000.png | HR26CE1485 | HR26CE1485 | 100.0% | ✓ | 0.495 |
| car-wbs-HR26CH3604_00000.jpeg | HR26CH3604 | HR26CH3604 | 100.0% | ✓ | 0.476 |
| car-wbs-HR26CK8571_00000.jpeg | HR26CK8571 | HR26CK8571 | 100.0% | ✓ | 0.401 |
| car-wbs-HR26CT4063_00000.jpeg | HR26CT4063 | HR26CT4063 | 100.0% | ✓ | 0.530 |
| car-wbs-HR26CT6702_00000.jpeg | HR26CT6702 | HR26CT6702 | 100.0% | ✓ | 0.479 |
| car-wbs-HR26CU6799_00000.png | HR26CU6799 | R26CU6799 | 10.0% | ✓ | 0.480 |
| car-wbs-HR26DA0471_00000.jpeg | HR26DA0471 | HR26DA0471 | 100.0% | ✓ | 0.448 |
| car-wbs-HR26DA2330_00000.jpeg | HR26DA2330 | HR26DA2330 | 100.0% | ✓ | 0.388 |
| car-wbs-HR51AY9099_00000.jpg | HR51AY9099 | HR51AY909 | 90.0% | ✓ | 0.467 |
| car-wbs-HR99EX6037_00000.jpeg | HR99EX6037 | HR99EXCTEMP603 | 42.9% | ✓ | 0.402 |
| car-wbs-KA031351_00000.jpeg | KA031351 | KA031351 | 100.0% | ✓ | 0.288 |
| car-wbs-KA03NA8385_00000.png | KA03NA8385 | KA03NA8385 | 100.0% | ✓ | 0.474 |
| car-wbs-KA04MN3622_00000.jpg | KA04MN3622 | KAO4MN3622 | 90.0% | ✓ | 0.413 |
| car-wbs-KA18P2987_00000.jpg | KA18P2987 | 2987KA18P | 11.1% | ✓ | 0.598 |
| car-wbs-KA51MJ8156_00000.png | KA51MJ8156 | KA51MJ8156 | 100.0% | ✓ | 0.405 |
| car-wbs-KL05AK3300_00000.jpeg | KL05AK3300 | KLO5AK3300 | 90.0% | ✓ | 0.289 |
| car-wbs-KL07CB8599_00000.jpeg | KL07CB8599 | L07CB8596 | 0.0% | ✓ | 0.426 |
| car-wbs-KL10AV6342_00000.jpeg | KL10AV6342 | KL1GAV6342 | 90.0% | ✓ | 0.197 |
| car-wbs-KL10AW2111_00000.jpeg | KL10AW2111 | KL10AW2111 | 100.0% | ✓ | 0.359 |
| car-wbs-KL12G7531_00000.jpeg | KL12G7531 | KL12G7531 | 100.0% | ✓ | 0.355 |
| car-wbs-KL20K7561_00000.jpeg | KL20K7561 | KL20K7561 | 100.0% | ✓ | 0.518 |
| car-wbs-KL53E964_00000.jpeg | KL53E964 | KL53E964 | 100.0% | ✓ | 0.395 |
| car-wbs-KL54H369_00000.jpeg | KL54H369 | KL541H369 | 44.4% | ✓ | 0.367 |
| car-wbs-KL60N5344_00001.jpeg | KL60N5344 | 578SN09 | 11.1% | ✓ | 0.395 |
| car-wbs-MH01AR5274_00000.jpeg | MH01AR5274 | MH01587 | 40.0% | ✓ | 0.463 |
| car-wbs-MH01AV8866_00000.png | MH01AV8866 | H01AV899 | 0.0% | ✓ | 0.362 |
| car-wbs-MH03BS7778_00001.jpeg | MH03BS7778 | MHD3BS7778 | 90.0% | ✓ | 0.523 |
| car-wbs-MH03BS7778_00002.jpeg | MH03BS7778 | MHD3BS7778 | 90.0% | ✓ | 0.328 |
| car-wbs-MH04DW8351_00000.jpeg | MH04DW8351 | MH04DW8351IND | 76.9% | ✓ | 0.459 |
| car-wbs-MH12NE8922_00000.png | MH12NE8922 | MH12NE8922 | 100.0% | ✓ | 0.450 |
| car-wbs-MH14DX9937_00000.png | MH14DX9937 | MH14DX9937 | 100.0% | ✓ | 0.391 |
| car-wbs-MH14EH5819_00000.jpeg | MH14EH5819 | MH14EH5819 | 100.0% | ✓ | 0.380 |
| car-wbs-MH14EH7958_00000.jpeg | MH14EH7958 | MH14EH7958 | 100.0% | ✓ | 0.447 |
| car-wbs-MH14EP4660_00000.jpeg | MH14EP4660 | MH14EP4660 | 100.0% | ✓ | 0.384 |
| car-wbs-MH15BD8877_00000.png | MH15BD8877 | MH15BD8877 | 100.0% | ✓ | 0.404 |
| car-wbs-MH20BN3525_00000.jpeg | MH20BN3525 | MHI2OBN3525 | 18.2% | ✓ | 0.541 |
| car-wbs-MH20BN3525_00001.jpeg | MH20BN3525 | MH2OBN3525 | 90.0% | ✓ | 0.496 |
| car-wbs-MH20BY3665_00000.jpeg | MH20BY3665 | MH20BY3665 | 100.0% | ✓ | 0.487 |
| car-wbs-MH20BY4465_00000.jpeg | MH20BY4465 | MH20BY4465 | 100.0% | ✓ | 0.467 |
| car-wbs-MH20CS1938_00000.jpeg | MH20CS1938 | MH20CS1938 | 100.0% | ✓ | 0.459 |
| car-wbs-MH20CS1941_00000.jpeg | MH20CS1941 | MH20CS194 | 90.0% | ✓ | 0.467 |
| car-wbs-MH20CS1941_00001.jpeg | MH20CS1941 | MH20CS1941 | 100.0% | ✓ | 0.522 |
| car-wbs-MH20EE045_00000.jpeg | MH20EE045 | MH20EE045 | 100.0% | ✓ | 0.467 |
| car-wbs-MH20EE0943_00000.jpeg | MH20EE0943 | 094333 | 0.0% | ✓ | 0.518 |
| car-wbs-MH20EE7598_00001.jpeg | MH20EE7598 | MH20EE7598 | 100.0% | ✓ | 0.434 |
| car-wbs-MH20EJ0364_00000.png | MH20EJ0364 | MH20EJ0364 | 100.0% | ✓ | 0.412 |
| car-wbs-MH21V9926_00000.jpg | MH21V9926 | MH21V9926 | 100.0% | ✓ | 0.288 |
| car-wbs-MH46X9996_00000.jpg | MH46X9996 | MH46X9998ND | 72.7% | ✓ | 0.545 |
| car-wbs-PB08CX2959_00000.jpeg | PB08CX2959 | OPB08CX2959 | 0.0% | ✓ | 0.456 |
| car-wbs-RJ02CC0784_00000.jpg | RJ02CC0784 | RJ02CC078 | 90.0% | ✓ | 0.415 |
| car-wbs-TN07BU5427_00001.jpeg | TN07BU5427 | TN07BU5427 | 100.0% | ✓ | 0.454 |
| car-wbs-TN07BV5200_00000.jpeg | TN07BV5200 | TN07BV5200 | 100.0% | ✓ | 0.376 |
| car-wbs-TN19S4523_00000.png | TN19S4523 | TN19S4523 | 100.0% | ✓ | 0.362 |
| car-wbs-TN21AT0479_00000.jpg | TN21AT0479 | TN21AT0479 | 100.0% | ✓ | 0.429 |
| car-wbs-TN21AT0480_00000.jpg | TN21AT0480 | TN21AT0480T0L14951KAZENNISSAN49551 | 29.4% | ✓ | 0.445 |
| car-wbs-TN21AT0480_00001.jpg | TN21AT0480 | TN21AT0480 | 100.0% | ✓ | 0.384 |
| car-wbs-TN21AT0492_00000.jpg | TN21AT0492 | TN21AT0492KAIZENNISSAN | 45.5% | ✓ | 0.417 |
| car-wbs-TN21AU1153_00000.png | TN21AU1153 | TN21AU1153 | 100.0% | ✓ | 0.413 |
| car-wbs-TN21BY0166_00000.png | TN21BY0166 | TN21BY0166 | 100.0% | ✓ | 0.367 |
| car-wbs-TN21BZ0768_00000.png | TN21BZ0768 | TN21BZ0768 | 100.0% | ✓ | 0.444 |
| car-wbs-TN28BA9999_00000.jpeg | TN28BA9999 | TN28BA9999 | 100.0% | ✓ | 0.455 |
| car-wbs-TN37CR4019_00000.jpeg | TN37CR4019 | TN37CR4019 | 100.0% | ✓ | 0.317 |
| car-wbs-TN52U1580_00000.jpeg | TN52U1580 | TN52U1580 | 100.0% | ✓ | 0.409 |
| car-wbs-TN58AM1_00000.jpeg | TN58AM1 | TN58AM1 | 100.0% | ✓ | 0.482 |
| car-wbs-TN59BE0939_00000.jpg | TN59BE0939 | RRCARS | 0.0% | ✓ | 0.546 |
| car-wbs-TN74AH1413_00000.jpeg | TN74AH1413 | TN74AH1413 | 100.0% | ✓ | 0.497 |
| car-wbs-UP16CT2233_00000.jpg | UP16CT2233 | UP16CT2233 | 100.0% | ✓ | 0.329 |
| car-wbs-UP32FH2653_00000.jpg | UP32FH2653 | JP32FH2653 | 90.0% | ✓ | 0.453 |
| car-wbs-UP50AS4535_00000.jpg | UP50AS4535 | UP50AS453 | 90.0% | ✓ | 0.453 |
| car-wbs-WB74X7605_00000.jpeg | WB74X7605 | WB74X7605 | 100.0% | ✓ | 0.332 |
| cbd81ff4-fc2a-4cd5-bed6-0fef4c456caa___226616d1495986750-zoomcar-review-mahindra-scorpio-s4-09-pamban.jpg.jpeg | KA51AA3469 | KA51AA3469 | 100.0% | ✓ | 0.446 |
| cc80e5c6-2cb7-40f7-a072-3f4cb5a56c65___Dileep_Car_Innova_1.jpg.jpeg | KL07BF2007 | KL07BF2007 | 100.0% | ✓ | 0.355 |
| cd506968-8676-4df3-9150-d1c2e618f4c2___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_nissan-note-india-testing_827x510_81508219864.jpg | TN21TC31 | TN21TC31 | 100.0% | ✓ | 0.471 |
| cf011553-f802-4b11-99ec-6e03d731b646___new_2014-VW-Polo-facelift-spied-India-rear.jpg.jpeg | MH14TCF460 | MH14TCF460 | 100.0% | ✓ | 0.459 |
| d1115e64-987f-4a72-b9bb-853f67ca37b6___new_2014-Volkswagen-Polo-Review.jpg.jpeg | MH14EH7958 | MH14EH7958 | 100.0% | ✓ | 0.441 |
| d16041c1-831d-4cc9-8869-68be8b6e12d7___slider-new-2.jpg.jpeg | KL12G7531 | KL12G7531 | 100.0% | ✓ | 0.424 |
| d3d2cbaa-f24a-4cb7-a236-53fa514816b4___new_2017-Skoda-Rapid-Diesel-DSG-Review-6.jpg.jpeg | MH20EE045 | MH20EE045 | 100.0% | ✓ | 0.436 |
| d5ddd1e6-56ca-4991-994e-cb78fe836a47___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_2013-Nissan-Terrano-Review.jpg | RJ27TC0530 | RJ27TC0530 | 100.0% | ✓ | 0.450 |
| d779f6f0-5731-48ed-bd43-836a3f7f2013___product-500x5000.jpeg | KL01CA2555 | KL01CA2555 | 100.0% | ✓ | 0.420 |
| db4f467d-2d9e-4567-ab68-fce4a8689562___maxresdefault7.jpg.jpeg | HR26BU0375 | HR26BU0375 | 100.0% | ✓ | 0.438 |
| dc8bb609-7086-431b-9c8b-9520975dabd2___822716205_6_1080x720_maruti-suzuki-ciaz-make-year-2014-petrol-.jpg.jpeg | KL07CB8599 | L07CB8596 | 0.0% | ✓ | 0.389 |
| dfebb432-82cd-4154-b919-0d8be2c0d85a___5e4697c6f639f3f341b2feb170d72795af3fc998.jpg.jpeg | KL25B2001 | KL25B2001IND | 75.0% | ✓ | 0.437 |
| e1d7921f-5e58-46c7-b4f9-9fd4585508f1___IMG_9072.jpg.jpeg | GJ1KF1111 | 11GJKI | 0.0% | ✓ | 0.349 |
| e204c50d-b861-4f68-827f-575351d150b4___Maruti-Baleno-RS-4.jpg.jpeg | HR26DA2330 | HR26DA2330 | 100.0% | ✓ | 0.332 |
| e29b6eea-504a-4da2-8503-5bd3443f3bd8___maruti-suzuki-wagon-r-front-left-rim.jpg.jpeg | KA04ME9869 | KA04ME9869 | 100.0% | ✓ | 0.446 |
| e4907f73-1fbc-412c-a766-9c81bb31449c___2016-Chevrolet-TrailBlazer-India-Test-Drive-Review-e1442832290349.jpg.jpeg | GJ17TC214 | GJ17TC214 | 100.0% | ✓ | 0.407 |
| e53076d3-c3b7-47b8-8c63-30a08b8ad983___40213__Maruti_Suzuki_Swift--016.JPG.jpeg | HR26DK6475 | HR26DK6475IND | 76.9% | ✓ | 0.384 |
| e706cec0-783a-42bc-b934-64788481cd92___4-Innova-After-2.jpg.jpeg | MH04DW8351 | MH04DW8351IND | 76.9% | ✓ | 0.494 |
| e719f39b-e8dd-4fe0-9a41-71123932a486___226676869_4_1000x700_maruti-suzuki-wagon-r-duo-petrol-120000-kms-2007-year-cars_rev001.jpg.jpeg | KL07BF5000 | KL07BF5000DN | 83.3% | ✓ | 0.514 |
| e750b959-1b5a-41bf-bf64-d8099089e909___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano-Duster-launch-dare-pics-price-interiors-images-1.jpg | TN19TC91 | TH119CTC91 | 30.0% | ✓ | 0.418 |
| e77e5db3-2bc6-4223-a258-2334256be65e___new_205356-maruti-swift-torque-blue-vdi-ddis-glistening-rockstar-enters-home-img_7531.jpg.jpeg | TN38BW1139 | TN38BW1139A | 90.9% | ✓ | 0.334 |
| e7a86597-7251-4e9b-9469-cd873fe6736f___Honda-Jazz-Number-Plates-Designs.jpg.jpeg | TN42R2697 | L692 | 11.1% | ✓ | 0.189 |
| e82fc408-a209-4221-9f6d-742b4b9d4c70___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_848803268_2_1080x720_nissan-terrano-xl-d-plus-2014-diesel-upload-photos.jpg | UP32FH2653 | JP32FH2653 | 90.0% | ✓ | 0.361 |
| ec4bde57-11dd-4d59-96ae-b5d3b28afda5___Maruti-Suzuki-Celerio-Back-Number-Plates-Design.jpg.jpeg | KL10AW2814 | KL10AW2814 | 100.0% | ✓ | 0.451 |
| ed6ee171-691a-4563-a79d-4f47d8ce9f85___Maruti-Suzuki-SCross-Exterior-80675.jpg.jpeg | HPSX4000 | HPSX4000 | 100.0% | ✓ | 0.442 |
| ed9cd3b2-76a2-4cc8-9e4e-76d50ad1ff68___maruti-suzuki-wagon-r-hood-open-view.jpg.jpeg | KA51MJ2143 | A53 | 0.0% | ✓ | 0.540 |
| ef2dc97f-bc6d-49a3-9ddb-03f54f3a20e7___Skoda-Rapid-new-Exterior-84398.jpg.jpeg | MH20EE0943 | MH20EE0943 | 100.0% | ✓ | 0.470 |
| f20708c6-14cd-46ff-b8f1-6faef5fd8d2d___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano-20346.jpg | RJ27TC0530 | RJ27TC0530 | 100.0% | ✓ | 0.430 |
| f59d42f8-dc55-4b60-bb91-c2db3644f2ff___841309873_1_1080x720_mahindra-scorpio-vlx-airbags-bs-iii-2012-diesel-.jpg.jpeg | TN58AM1 | TN58AM1 | 100.0% | ✓ | 0.403 |
| f7621229-eebc-4dff-aef8-e79fac571be4___Skoda-Octavia-5.jpg.jpeg | MH20CS1938 | MH20CS1938 | 100.0% | ✓ | 0.365 |
| f94e9bcf-ec4a-418a-8685-14b71b82a039___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_848126147_1_1080x720_nissan-terrano-xl-d-plus-2014-diesel-surat.jpg | GJ05JH2501 | GJ05JH250 | 90.0% | ✓ | 0.435 |
| f9e27b82-9e65-44ba-bf9a-b3ced8901937___Skoda-Superb-Exterior-deadfront-69204.jpg.jpeg | MH20DV2362 | MH20DV2362 | 100.0% | ✓ | 0.425 |
| fac9f4e6-c4b9-4872-86da-79d58ea2adc5___5273.jpg.jpeg | MH02B16890 | 6890 | 0.0% | ✓ | 0.420 |
| fb939bb1-5257-4879-bafb-539472ffb73d___Maruti-Suzuki-Brezza-Number-Plates-Design.jpg.jpeg | KL10AW2111 | KL10AW2111 | 100.0% | ✓ | 0.429 |
| fbc107b7-13d8-4a2d-b8be-7c96bd7d609c___new_9214a754343030395508e104d7948167_large.jpg.jpeg | PB08CX2959 | PB08CX2959 | 100.0% | ✓ | 0.485 |
| fc1f58f0-2f54-4545-8915-2936d08dcce1___new_Skoda_Laura_DSG_Long_Term_Review-125.jpg.jpeg | MH20BN3525 | MH2OBN3525 | 90.0% | ✓ | 0.334 |
| fda5760a-c6ff-43a9-8ce7-61e808f109ea___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_nissan_terrano_ii_nissan_terrano_ii_2_7_tdi_se_4x4_3330085523722771900.jpg | HD02NP0 | HD02NP | 85.7% | ✓ | 0.367 |

## Files With No OCR Text

| Image | Ground Truth | Time (s) |
|-------|--------------|----------|
| 18f2e55e-0724-4eee-a28e-5d552a5aa045___20455d0e2dca458f13fbf4da5a2dc118.jpg - Copy.jpeg | HP896786 | 0.138 |
| 1e7151e6-8626-4fb2-b7e8-0fb6a8fde713___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_a19be5870f8f2593c994c465a691477e_555X416_1.jpg | RJ27TC0530 | 0.387 |
| 1f0a7fe8-ac9b-4b54-9d55-e11f34f9f98b___9f6521b2ddb7b8e40ffd510966ffcca3.jpg.jpeg | KA42TC131011 | 0.424 |
| 28574c06-ac11-422e-864f-a51b726bb9ba___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_hqdefault0.jpg | RJ27TC0530 | 0.229 |
| 38cee3dd-dead-4ba0-b350-0f82d94836da___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_big_467860_1505633598.jpg | HR26TC7303 | 0.520 |
| 3fd93f90-f778-4f14-8b0f-67dd0be88a06___561644d1308248087-beware-fake-registration-numbers-imported-cars-proof-pg-3-257491_10150223034216705_685851704_7325894_6829197_o.jpg.jpeg | WB06F9209 | 0.365 |
| 41209caa-2aa2-4003-b057-e0508c1d5ac9___Volkswagen-Ameo-Exterior-68165.jpg.jpeg | MH14TCD204 | 0.406 |
| 44ee098a-c33c-47a2-9dcb-717c02d5a240___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_2013-Nissan-Terrano-Front.jpg | TN19TC91 | 0.403 |
| 59a29a85-30f5-469e-a6c2-e1af171f151c___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_10187746494_417a40a489 - Copy.jpg | TN21AU7234 | 0.417 |
| 5a4780bf-d2af-4f19-a36b-65004cd06b3d___maruti-suzuki-wagon-r-rear-left-rim.jpg.jpeg | MP09CC1667 | 0.474 |
| 65e2825e-943f-48a3-a4e2-7e81d649ace2___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_nissan-terrano-L.jpg | RJ27TC0530 | 0.400 |
| 6cd15e75-3e26-41f2-a6aa-b1b58dec1c48___volkswagen-polo-exquisite-limited-edition-m1_720x540.jpg.jpeg | MH14EY5972 | 0.372 |
| 6e52e238-927b-46e0-a8f1-5f77ef1ecf9f___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_nissan-terrano-rear.jpg | RJ27TC0530 | 0.281 |
| 7506a48d-2c70-4b12-b300-463754bc7837___165979d1427773878-new-volkswagen-vento-2016-launched-india-vw-vento-facelift-rear-india-spotted-test.jpg.jpeg | MH14TCF459 | 0.136 |
| 84969111-2ec3-4914-b3bb-da5aaac701e7___1.jpg.jpeg | MH20TC830C | 0.458 |
| 85bd88a0-ac9a-412e-ba44-2a29d209e212___4336_1466054535_9.jpg.jpeg | MH14TCE4 | 0.363 |
| 868b996f-1f40-47ca-8222-b0f2750d7010___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_nissan-terrano-left-side-view.jpg | AS23M1264 | 0.377 |
| ae0bb069-0e7e-412a-9473-e8e0d4041cbc___135718d1400912281-skoda-rapid-1-6tdi-cr-mt-elegance-ultima-candy-white-white-monster-_mg_3000comp.jpg.jpeg | TN74F3339 | 0.407 |
| aea95aa0-a000-4103-945d-c173383bd646___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_Nissan-Terrano-11.jpg | RJ27TC0530 | 0.358 |
| af058398-67d6-4d0e-a6d9-afc1cb184e5b___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_hqdefault3.jpg | RJ27TC0530 | 0.300 |
| b79f870d-15c7-48e3-8874-7c0e53e29f75___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_754eab882b8290b29a0454dd45580a39_555X416_1.jpg | RJ27TC0530 | 0.437 |
| ba457a81-cf10-4879-8e8f-3a0a8d10689a___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_hqdefault.jpg | RJ27TC0530 | 0.526 |
| bb767d81-73b4-4c47-b619-7efde490b199___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_oem-name0 - Copy.jpg | RJ27TC0530 | 0.341 |
| bbf22d00-3dd6-4338-b701-264e0c0c0aff___Maruti-Swift-DZire-Photos-1.jpg.jpeg | HR26CM6005 | 0.404 |
| bff789cd-2d58-4399-a55e-33be55f313e3___IMG_0472.jpg.jpeg | MH02CT2727 | 0.384 |
| c48f8704-bc40-462a-a595-445b2f2e63fe___new_77cca4b6f3864b3928824a030ba101d2.jpg.jpeg | KL60N5344 | 0.420 |
| car-wbs-HR26BU0380_00000.png | HR26BU0380 | 0.465 |
| car-wbs-HR26CM6005_00000.jpeg | HR26CM6005 | 0.549 |
| car-wbs-MH20CS4946_00000.jpeg | MH20CS4946 | 0.543 |
| car-wbs-MH20EE7597_00000.jpeg | MH20EE7597 | 0.126 |
| car-wbs-TN21BZ0768_00001.png | TN21BZ0768 | 0.319 |
| car-wbs-TN38BY4191_00000.jpeg | TN38BY4191 | 0.480 |
| car-wbs-TN74F3339_00000.jpeg | TN74F3339 | 0.371 |
| d9023f53-f90d-4b98-bd52-30796777ddcf___Maruti-Suzuki-Baleno-hatchback-56545.jpg.jpeg | HR26TC5656 | 0.433 |
| e54fb93b-020f-4c55-b4db-86e17ba9b9e0___2014-VW-Polo-facelift-rear-spotted-testing-in-Pune-India.jpg.jpeg | MH14TCP237 | 0.460 |
| f5bbb8da-dddd-47ae-a46c-15492f9cf752___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_10187746544_0c0822790e - Copy.jpg | TN21AU7234 | 0.430 |
| f7ba7a93-e8d7-4253-b327-d87e3f2bb747___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_2013-Nissan-Terrano-Rear.jpg | TN19TC94 | 0.459 |
| fac9f4e6-c4b9-4872-86da-79d58ea2adc5___5273.jpg - Copy.jpeg | UP16AB3726 | 0.508 |

---
*Report generated on 2026-04-03T15:48:46.233184*
