# Triton Model Files - Complete Copy

## Table of Contents
- [anpr_pipeline](#anpr_pipeline)
  - [model.py](#anpr_pipeline-modelpy)
  - [config.pbtxt](#anpr_pipeline-configpbtxt)
- [paddle_ocr_gpu_batched](#paddle_ocr_gpu_batched)
  - [model.py](#paddle_ocr_gpu_batched-modelpy)
  - [config.pbtxt](#paddle_ocr_gpu_batched-configpbtxt)
- [docker-compose.yml](#docker-composeyml)

---

## anpr_pipeline

### anpr_pipeline/model.py

```python
import json
import re
import time
import math
import base64
import cv2
import numpy as np
import concurrent.futures
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the model configuration.
        Triton calls this once when the model is loaded.
        """
        # Set your exact model names as they appear in the model_repository
        self.vehicle_model = "vehicle_detection_rt_detr"
        self.plate_model = "plate_region_detection_rt_detr"
        self.ocr_model = "paddle_ocr_gpu_batched"

        self.vehicle_input_name = "images"
        self.vehicle_output_name = "output0"
        self.plate_input_name = "images"
        self.plate_output_name = "output0"
        self.ocr_input_name = "INPUT1"
        self.ocr_output_name = "OUTPUT0"

        self.vehicle_class_id_name_map = {
            0: "animal",
            1: "autorickshaw",
            2: "bicycle",
            3: "bus",
            4: "car",
            5: "caravan",
            6: "motorcycle",
            7: "person",
            8: "rider",
            9: "traffic light",
            10: "traffic sign",
            11: "trailer",
            12: "train",
            13: "truck",
            14: "vehicle fallback",
        }
        self.plate_candidate_vehicle_classes = {
            "autorickshaw",
            "bicycle",
            "bus",
            "car",
            "caravan",
            "motorcycle",
            "truck",
            "vehicle fallback",
        }

        ### Vehicle
        # Detection confidence threshold to keep low-confidence vehicles
        self.vehicle_conf_threshold = 0.4
        # Confidence threshold to keep weak plate detections
        self.plate_conf_threshold = 0.4
        ### Detector
        # Target size for detector inputs
        self.detector_input_size = (640, 640)
        ### Plate validation
        ### OCR crop padding
        # Horizontal padding ratio applied when expanding the plate bbox for OCR crop extraction
        self.ocr_crop_padding_ratio_x = 0.08
        # Vertical padding ratio applied when expanding the plate bbox for OCR crop extraction
        self.ocr_crop_padding_ratio_y = 0.20
        ### Logging
        # Border width for log messages
        self.log_border_width = 150
        self.log_border = "-" * self.log_border_width
        ### OCR quality
        # Minimum Laplacian variance required before sending a crop to OCR
        self.ocr_min_laplacian_variance = 50.0
        # Plate crop width constraints (pixels)
        self.ocr_min_width = 30
        self.ocr_max_width = 300
        # Plate crop height constraints (pixels)
        self.ocr_min_height = 12
        self.ocr_max_height = 150

        self.sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        # FIX: Two separate LUTs for correct lighting adjustment
        # Darken overexposed images (> 1.0)
        self.lut_darken = np.array([((i / 255.0) ** 1.5) * 255 for i in range(256)]).astype("uint8")

        # Brighten underexposed images (< 1.0)
        self.lut_brighten = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)]).astype("uint8")

        self.clahe_strong = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.clahe_light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.cpu_memory = pb_utils.PreferredMemory(
            pb_utils.TRITONSERVER_MEMORY_CPU, 0
        )

        try:
            config = json.loads(args.get("model_config", "{}"))
        except json.JSONDecodeError:
            config = {}
        instance_groups = config.get("instance_group", [])
        instance_descriptions = []
        for group in instance_groups:
            kind = group.get("kind", "KIND_UNKNOWN")
            gpus = group.get("gpus", [])
            instance_descriptions.append(f"{kind}:{gpus if gpus else 'auto'}")
        if instance_descriptions:
            pb_utils.Logger.log_warn(
                f"[ANPR_PIPELINE] [profile] anpr_pipeline instance configuration: {', '.join(instance_descriptions)}"
            )
        else:
            pb_utils.Logger.log_warn("[ANPR_PIPELINE] [profile] anpr_pipeline instance configuration unavailable")

    def execute(self, requests):
        """
        Executes the pipeline for every incoming request.
        """
        responses = []
        for request_index, request in enumerate(requests):
            execute_start = time.perf_counter()
            in_tensor = pb_utils.get_input_tensor_by_name(request, "image")
            img_numpy = in_tensor.as_numpy()
            batch_images = self._prepare_batch_images(img_numpy)
            batch_size = len(batch_images)
            batch_results = []
            request_error = None
            request_id = request.request_id()

            start_msg = f"ANPR EXEC START | Request ID: {request_id if request_id is not None else '-'}"
            pb_utils.Logger.log_info(f"{self.log_border}")
            pb_utils.Logger.log_info(f"{start_msg}")
            pb_utils.Logger.log_info(f"{self.log_border}")

            try:
                batch_results = self._run_batched_pipeline(batch_images, request_id=request_id)
            except ValueError as exc:
                request_error = pb_utils.TritonError(str(exc))
                pb_utils.Logger.log_error(
                    "[ANPR_PIPELINE] anpr_pipeline execute error: "
                    f"request_index={request_index} "
                    f"request_id={request_id if request_id is not None else '-'} "
                    f"error={exc}"
                )
            except Exception as exc:
                request_error = pb_utils.TritonError(str(exc))
                pb_utils.Logger.log_error(
                    "[ANPR_PIPELINE] anpr_pipeline execute error: "
                    f"request_index={request_index} "
                    f"request_id={request_id if request_id is not None else '-'} "
                    f"error={exc}"
                )

            if request_error is not None:
                duration_ms = (time.perf_counter() - execute_start) * 1000.0
                end_msg = f"ANPR EXEC END | Request: {request_index} | ID: {request_id if request_id is not None else '-'} | Total: {duration_ms:.2f}ms"
                pb_utils.Logger.log_info(f"{self.log_border}")
                pb_utils.Logger.log_info(f"{end_msg}")
                pb_utils.Logger.log_info(f"{self.log_border}")
                responses.append(pb_utils.InferenceResponse(error=request_error))
                continue

            out_tensor = pb_utils.Tensor("anpr_results", np.array(batch_results, dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

            duration_ms = (time.perf_counter() - execute_start) * 1000.0
            end_msg = f"ANPR EXEC END | Request ID: {request_id if request_id is not None else '-'} | Total: {duration_ms:.2f}ms"
            pb_utils.Logger.log_info(f"{self.log_border}")
            pb_utils.Logger.log_info(f"{end_msg}")
            pb_utils.Logger.log_info(f"{self.log_border}")

        return responses

    # ---------------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------------

    def _run_batched_pipeline(self, batch_images, request_id=None):
        total_start = time.perf_counter()

        # 1. Preprocess full frames for vehicle detection
        batch_tensor, batch_meta = self._preprocess_for_detector(batch_images)
        preprocess_ms = (time.perf_counter() - total_start) * 1000.0
        _, _, img_h, img_w = batch_tensor.shape

        # 2. Vehicle Detection in Batch Mode
        vehicle_infer_start = time.perf_counter()
        veh_request = pb_utils.InferenceRequest(
            model_name=self.vehicle_model,
            requested_output_names=[self.vehicle_output_name],
            inputs=[pb_utils.Tensor(self.vehicle_input_name, batch_tensor)],
            preferred_memory=self.cpu_memory,
            request_id=request_id,
        )
        veh_response = veh_request.exec()
        vehicle_ms = (time.perf_counter() - vehicle_infer_start) * 1000.0

        if veh_response.has_error():
            raise pb_utils.TritonModelException(veh_response.error().message())

        veh_output = pb_utils.get_output_tensor_by_name(
            veh_response, self.vehicle_output_name
        ).as_numpy()

        # Parse vehicle detections for each frame
        batch_vehicles = []
        for batch_idx, original_bgr in enumerate(batch_images):
            valid_vehicles = self._parse_rt_detr(
                veh_output[batch_idx: batch_idx + 1],
                original_bgr.shape[1],
                original_bgr.shape[0],
                self.vehicle_conf_threshold,
                preprocess_meta=batch_meta[batch_idx] if batch_meta else None
            )
            batch_vehicles.append(valid_vehicles)

        # 3. Cascaded Plate Detection on Vehicle Crops
        plate_infer_start = time.perf_counter()

        # Identify eligible vehicles for plate detection across the whole batch
        eligible_vehicle_crops = []
        for batch_idx, frame_vehicles in enumerate(batch_vehicles):
            original_bgr = batch_images[batch_idx]
            for veh_idx, vehicle in enumerate(frame_vehicles):
                vehicle_class_name = str(vehicle.get("class_name", "")).strip().lower()
                vehicle_conf = vehicle.get("conf", 0.0)
                # Filter by class name AND confidence threshold
                if vehicle_class_name in self.plate_candidate_vehicle_classes and vehicle_conf >= self.vehicle_conf_threshold:
                    # Crop the vehicle
                    vx1, vy1, vx2, vy2 = vehicle["bbox"]
                    vehicle_crop = original_bgr[vy1:vy2, vx1:vx2]
                    if vehicle_crop.size > 0:
                        # pb_utils.Logger.log_info(
                        #     f"anpr_pipeline cascading: cropping vehicle {vehicle_class_name} "
                        #     f"at {vehicle['bbox']} -> shape {vehicle_crop.shape}"
                        # )
                        eligible_vehicle_crops.append({
                            "frame_idx": batch_idx,
                            "veh_idx": veh_idx,
                            "crop": vehicle_crop,
                            "offset": (vx1, vy1),
                            "vehicle": vehicle
                        })

        all_valid_plates = []  # will store mapped plates: {frame_idx, veh_idx, plates: []}
        plate_ms = 0.0
        if eligible_vehicle_crops:
            # Batch preprocess vehicle crops for plate detector
            crop_images = [item["crop"] for item in eligible_vehicle_crops]
            crop_tensor, crop_meta = self._preprocess_for_detector(crop_images)
            plate_exec_start = time.perf_counter()

            pb_utils.Logger.log_verbose(
                f"[ANPR_PIPELINE] [plate_detection] eligible_vehicles={len(crop_images)}"
            )

            plate_output_chunks = []

            # Plate Detection in Batch Mode on Vehicle Crops (chunked to max_batch_size)
            max_plate_batch_size = 24
            num_chunks = math.ceil(len(crop_images) / max_plate_batch_size)
            pb_utils.Logger.log_verbose(
                f"[ANPR_PIPELINE] [plate_detection] sending {len(crop_images)} vehicles in {num_chunks} chunk(s)"
            )
            for i in range(0, len(crop_images), max_plate_batch_size):
                chunk_tensor = crop_tensor[i:i + max_plate_batch_size]
                chunk_size = len(chunk_tensor)
                pb_utils.Logger.log_verbose(
                    f"[ANPR_PIPELINE] [plate_detection] chunk {i // max_plate_batch_size + 1}/{num_chunks} with {chunk_size} vehicles"
                )
                plate_request = pb_utils.InferenceRequest(
                    model_name=self.plate_model,
                    requested_output_names=[self.plate_output_name],
                    inputs=[pb_utils.Tensor(self.plate_input_name, chunk_tensor)],
                    preferred_memory=self.cpu_memory,
                    request_id=request_id,
                )
                plate_response = plate_request.exec()

                if plate_response.has_error():
                    raise pb_utils.TritonModelException(plate_response.error().message())

                chunk_output = pb_utils.get_output_tensor_by_name(
                    plate_response, self.plate_output_name
                ).as_numpy()
                plate_output_chunks.append(chunk_output)

            plate_ms = (time.perf_counter() - plate_exec_start) * 1000.0

            if plate_output_chunks:
                plate_output = np.concatenate(plate_output_chunks, axis=0)
            else:
                plate_output = np.array([])

            # Parse plate detections for each vehicle crop
            for crop_idx, crop_item in enumerate(eligible_vehicle_crops):
                crop_h, crop_w = crop_item["crop"].shape[:2]
                valid_plates = self._parse_plate_output_full_frame(
                    plate_output[crop_idx: crop_idx + 1],
                    crop_w,
                    crop_h,
                    self.plate_conf_threshold,
                    preprocess_meta=crop_meta[crop_idx]
                )

                # Map plate coordinates from crop-space back to full-frame-space
                ox, oy = crop_item["offset"]
                mapped_plates = []
                for plate in valid_plates:
                    px1, py1, px2, py2 = plate["bbox"]
                    mapped_plate = {
                        "bbox": [px1 + ox, py1 + oy, px2 + ox, py2 + oy],
                        "bbox_in_vehicle": [px1, py1, px2, py2],
                        "conf": float(plate.get("conf", 0.0)),
                        "class_id": int(plate.get("class_id", 0)),
                    }
                    mapped_plates.append(mapped_plate)

                crop_item["detected_plates"] = mapped_plates
        else:
            plate_ms = 0.0

        # 4. Assemble Final Detections and Queue for OCR
        batch_final_detections = []
        all_pending_ocr_items = []
        total_vehicle_count = 0
        total_plate_count = 0
        valid_ocr_count = 0
        plate_detection_count = 0
        matching_ms = 0.0  # Matching is now implicit
        ocr_total_ms = 0.0
        ocr_call_count = 0
        ocr_preprocess_total_ms = 0.0

        # Organize detected plates by frame and vehicle
        plates_by_veh = {}  # key: (frame_idx, veh_idx), value: [plates]
        for item in eligible_vehicle_crops:
            if "detected_plates" in item and item["detected_plates"]:
                plates_by_veh[(item["frame_idx"], item["veh_idx"])] = item["detected_plates"]

        for batch_idx, original_bgr in enumerate(batch_images):
            frame_final_detections = []
            frame_vehicles = batch_vehicles[batch_idx]
            total_vehicle_count += len(frame_vehicles)

            # Keep track of plates processed in this frame to deduplicate globally
            # across overlapping vehicle crops
            processed_frame_plate_bboxes = []
            frame_matched_plate_count = 0

            for veh_idx, vehicle in enumerate(frame_vehicles):
                vehicle_class_name = str(vehicle.get("class_name", "")).strip().lower()
                vehicle_payload = {
                    "vehicle_bbox": vehicle["bbox"],
                    "vehicle_confidence": float(vehicle["conf"]),
                    "vehicle_class": str(vehicle.get("class_name", "unknown")),
                    "vehicle_class_id": int(vehicle.get("class_id", -1)),
                    "plates": []
                }

                # Collect OCR status messages for this vehicle
                ocr_status_messages = []

                if vehicle_class_name not in self.plate_candidate_vehicle_classes:
                    # Log non-candidate vehicle in single line
                    pb_utils.Logger.log_verbose(
                        f"[ANPR_PIPELINE] anpr_pipeline vehicle: class='{vehicle_class_name}' conf={float(vehicle['conf']):.3f} bbox={vehicle_payload['vehicle_bbox']} | plates=0 | OCR: []"
                    )
                    frame_final_detections.append(vehicle_payload)
                    continue

                # Get plates detected for this vehicle crop
                matched_plates = plates_by_veh.get((batch_idx, veh_idx), [])

                # First deduplicate within the vehicle crop itself
                matched_plates = self._deduplicate_plates(matched_plates)

                # Global deduplication across all vehicles in the current frame
                unique_matched_plates = []
                for plate in matched_plates:
                    is_global_duplicate = False
                    for processed_bbox in processed_frame_plate_bboxes:
                        if self._bbox_iou(plate["bbox"], processed_bbox) > 0.6:
                            is_global_duplicate = True
                            break

                    if not is_global_duplicate:
                        unique_matched_plates.append(plate)
                        processed_frame_plate_bboxes.append(plate["bbox"])

                matched_plates = unique_matched_plates

                vx1, vy1, vx2, vy2 = vehicle["bbox"]
                vehicle_crop_for_ocr = original_bgr[vy1:vy2, vx1:vx2]
                vehicle_crop_h, vehicle_crop_w = vehicle_crop_for_ocr.shape[:2] if vehicle_crop_for_ocr.size > 0 else (
                    0, 0)

                for plate in matched_plates:
                    plate_bbox = plate["bbox"]
                    plate_conf = plate.get("conf", 0.0)

                    plate_crop = np.empty((0, 0, 3), dtype=np.uint8)
                    plate_bbox_in_vehicle = plate.get("bbox_in_vehicle")
                    if plate_bbox_in_vehicle and vehicle_crop_for_ocr.size > 0:
                        expanded_plate_bbox_in_vehicle = self._expand_bbox(
                            plate_bbox_in_vehicle,
                            vehicle_crop_w,
                            vehicle_crop_h,
                            margin_ratio_x=self.ocr_crop_padding_ratio_x,
                            margin_ratio_y=self.ocr_crop_padding_ratio_y,
                        )
                        plate_crop = self._crop_original_plate(vehicle_crop_for_ocr, expanded_plate_bbox_in_vehicle)

                    if plate_crop.size == 0:
                        expanded_plate_bbox = self._expand_bbox(
                            plate_bbox,
                            original_bgr.shape[1],
                            original_bgr.shape[0],
                            margin_ratio_x=self.ocr_crop_padding_ratio_x,
                            margin_ratio_y=self.ocr_crop_padding_ratio_y,
                        )
                        plate_crop = self._crop_original_plate(original_bgr, expanded_plate_bbox)

                    if plate_crop.size == 0:
                        continue

                    plate_payload = {
                        "plate_bbox": plate_bbox,
                        "plate_confidence": float(plate_conf),
                        "text": "",
                        "text_confidence": 0.0,
                    }
                    vehicle_payload["plates"].append(plate_payload)

                    # Check quality before queuing for OCR
                    t0_start = time.perf_counter()
                    ocr_ready_crop, ocr_status = self._prepare_plate_crop_for_ocr(plate_crop)
                    t1_prepare = (time.perf_counter() - t0_start) * 1000.0  # Convert to ms
                    ocr_preprocess_total_ms += t1_prepare
                    if ocr_status:
                        ocr_status_messages.append(ocr_status)
                    if ocr_ready_crop is not None:
                        pb_utils.Logger.log_verbose(
                            f"[ANPR_PIPELINE] [OCR INPUT] frame={batch_idx} vehicle_bbox={vehicle['bbox']} plate_bbox={plate_bbox} crop_shape={ocr_ready_crop.shape}"
                        )
                        all_pending_ocr_items.append(
                            {
                                "plate_crop": ocr_ready_crop,
                                "plate_payload": plate_payload,
                                "vehicle_bbox": vehicle["bbox"],
                                "plate_bbox": plate_bbox,
                                "frame_idx": batch_idx,  # track which frame it belongs to
                                "ocr_idx": len(all_pending_ocr_items),  # unique index for sequence tracking
                            }
                        )

                # Log merged vehicle info in single line
                matched_count = len(matched_plates)
                if matched_count > 0:
                    total_plate_count += matched_count
                    frame_matched_plate_count += matched_count

                ocr_summary = ", ".join(ocr_status_messages) if ocr_status_messages else "[]"
                pb_utils.Logger.log_verbose(
                    f"[ANPR_PIPELINE] anpr_pipeline vehicle: class='{vehicle_class_name}' conf={float(vehicle['conf']):.3f} bbox={vehicle_payload['vehicle_bbox']} | plates={matched_count} | OCR: [{ocr_summary}]"
                )

                frame_final_detections.append(vehicle_payload)

            batch_final_detections.append(frame_final_detections)

            # Find ocr candidates for this frame
            frame_ocr_count = sum(1 for item in all_pending_ocr_items if item.get("frame_idx") == batch_idx)

            pb_utils.Logger.log_verbose(
                "[ANPR_PIPELINE] anpr_pipeline frame stats: "
                f"request_id={request_id if request_id is not None else '-'} "
                f"frame_index={batch_idx} "
                f"vehicles={len(frame_vehicles)} "
                f"plates={frame_matched_plate_count} "
                f"matched_plates={frame_matched_plate_count} "
                f"ocr_candidates={frame_ocr_count}"
            )

        # Run OCR over all accumulated items from the entire batch
        if all_pending_ocr_items:
            num_items = len(all_pending_ocr_items)
            pb_utils.Logger.log_verbose(
                f"[ANPR_PIPELINE] anpr_pipeline invoking OCR for {num_items} plates in batch mode across {len(batch_images)} images"
            )
            # Determine chunk size to maximize distribution across GPU instances
            num_instances = 1
            max_batch_size = 96
            chunk_size = min(max_batch_size, math.ceil(num_items / num_instances))
            if chunk_size == 0:
                chunk_size = 1

            chunks = []
            for i in range(0, num_items, chunk_size):
                chunks.append(all_pending_ocr_items[i: i + chunk_size])

            def process_ocr_chunk(chunk):
                crops = [self._normalize_plate_crop_for_ocr(item["plate_crop"]) for item in chunk]
                # for item, crop in zip(chunk, crops):
                #     pb_utils.Logger.log_warn(
                #         f"[OCR DEBUG] vehicle_bbox={item.get('vehicle_bbox')} plate_bbox={item.get('plate_bbox')} crop_shape={crop.shape}"
                #     )
                #     self._image_to_base64(crop)
                batched_input = self._pad_and_batch_ocr_inputs(crops)
                ocr_req = pb_utils.InferenceRequest(
                    model_name=self.ocr_model,
                    requested_output_names=[self.ocr_output_name],
                    inputs=[pb_utils.Tensor(self.ocr_input_name, batched_input)],
                    preferred_memory=self.cpu_memory,
                    request_id=request_id,
                )
                start_t = time.perf_counter()
                ocr_response = ocr_req.exec()
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                # Return indices along with chunk for sequence tracking
                indices = [item["ocr_idx"] for item in chunk]
                return indices, ocr_response, elapsed_ms

            ocr_start = time.perf_counter()
            # Use OCR instance count to optimize parallelization (default to 1)
            max_workers = min(len(chunks), 1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_ocr_chunk, chunk) for chunk in chunks]
                for future in concurrent.futures.as_completed(futures):
                    indices, ocr_response, elapsed_ms = future.result()
                    ocr_total_ms += elapsed_ms
                    ocr_call_count += 1

                    if ocr_response.has_error():
                        pb_utils.Logger.log_error(
                            f"[ANPR_PIPELINE] anpr_pipeline OCR inference error: {ocr_response.error().message()}"
                        )
                        continue

                    ocr_raw_output = pb_utils.get_output_tensor_by_name(
                        ocr_response, self.ocr_output_name
                    ).as_numpy()

                    ocr_results = self._parse_ocr_outputs(ocr_raw_output)
                    # Match results by index to guarantee sequence preservation
                    for ocr_idx, ocr_res in zip(indices, ocr_results):
                        ocr_text, ocr_conf = ocr_res
                        # Find the item with matching ocr_idx
                        for item in all_pending_ocr_items:
                            if item["ocr_idx"] == ocr_idx:
                                item["plate_payload"]["text"] = str(ocr_text)
                                item["plate_payload"]["text_confidence"] = float(ocr_conf)
                                if ocr_text:  # Only count non-empty results
                                    valid_ocr_count += 1
                                # pb_utils.Logger.log_info(
                                #     f"[OCR RESULT] ocr_idx={ocr_idx}, text='{ocr_text}', conf={ocr_conf:.3f}"
                                # )
                                break

            rejected_count = sum(1 for item in all_pending_ocr_items if not item["plate_payload"]["text"])
            pb_utils.Logger.log_verbose(
                f"[ANPR_PIPELINE] anpr_pipeline OCR batch inference finished: {ocr_call_count} batch(es) completed in {(time.perf_counter() - ocr_start) * 1000.0:.2f}ms | preprocess={ocr_preprocess_total_ms:.2f}ms | plates_sent={len(all_pending_ocr_items)} | valid_ocr={valid_ocr_count} | rejected={rejected_count}"
            )

        # Count plate detections (plates that didn't get OCR results)
        plate_detection_count = sum(1 for item in all_pending_ocr_items if not item["plate_payload"]["text"])

        # Log plates without OCR results
        # for item in all_pending_ocr_items:
        #     if not item["plate_payload"]["text"]:
        #         pb_utils.Logger.log_info(
        #             f"[OCR NO RESULT] ocr_idx={item['ocr_idx']}, text='', conf=0.000"
        #         )

        # Assemble the final batch_results
        batch_results = []
        for detections in batch_final_detections:
            batch_results.append(json.dumps({
                "status": "success",
                "detections": detections
            }))

        total_ms = (time.perf_counter() - total_start) * 1000.0
        pb_utils.Logger.log_warn(
            "[ANPR_PIPELINE] [profile] anpr_pipeline batch timings: "
            f"batch_size={len(batch_images)}, "
            f"preprocess_ms={preprocess_ms:.2f}, "
            f"vehicle_ms={vehicle_ms:.2f}, "
            f"plate_ms={plate_ms:.2f}, "
            f"matching_ms={matching_ms:.2f}, "
            f"ocr_preprocess_ms={ocr_preprocess_total_ms:.2f}, "
            f"ocr_total_ms={ocr_total_ms:.2f}, "
            f"ocr_calls={ocr_call_count}, "
            f"vehicles={total_vehicle_count}, plates={total_plate_count}, "
            f"valid_ocr={valid_ocr_count}, plate_detection_count={plate_detection_count}, "
            f"total_ms={total_ms:.2f}"
        )

        return batch_results

    def _prepare_batch_images(self, img_numpy):
        if img_numpy.ndim == 3:
            return [self._prepare_original_image(img_numpy)]
        if img_numpy.ndim != 4:
            raise ValueError(f"Unsupported input shape for image tensor: {img_numpy.shape}")
        return [self._prepare_original_image(single_image) for single_image in img_numpy]

    def _prepare_original_image(self, img_numpy):
        if img_numpy.ndim != 3:
            raise ValueError(f"Unsupported input shape for image tensor: {img_numpy.shape}")

        if img_numpy.shape[0] in (1, 3) and img_numpy.shape[-1] not in (1, 3):
            img_numpy = np.transpose(img_numpy, (1, 2, 0))

        if img_numpy.ndim != 3 or img_numpy.shape[2] not in (1, 3):
            raise ValueError(f"Expected HWC image with 1 or 3 channels, got shape: {img_numpy.shape}")

        if img_numpy.shape[2] == 1:
            img_numpy = np.repeat(img_numpy, 3, axis=2)

        if np.issubdtype(img_numpy.dtype, np.floating):
            max_value = float(np.max(img_numpy)) if img_numpy.size > 0 else 0.0
            if max_value <= 1.0:
                img_numpy = np.clip(img_numpy * 255.0, 0.0, 255.0)
            else:
                img_numpy = np.clip(img_numpy, 0.0, 255.0)
            img_numpy = img_numpy.astype(np.uint8)
        else:
            img_numpy = np.clip(img_numpy, 0, 255).astype(np.uint8)
        return img_numpy

    def _preprocess_for_detector(self, images):
        batch = []
        batch_meta = []
        for image in images:
            original_height, original_width = image.shape[:2]
            scale = min(self.detector_input_size[1] / original_width, self.detector_input_size[0] / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            resized = cv2.resize(image, (new_width, new_height))
            padded = np.full(
                (self.detector_input_size[0], self.detector_input_size[1], 3),
                114,
                dtype=np.uint8,
            )

            y_offset = (self.detector_input_size[0] - new_height) // 2
            x_offset = (self.detector_input_size[1] - new_width) // 2
            padded[y_offset: y_offset + new_height, x_offset: x_offset + new_width] = resized
            batch.append(padded)
            batch_meta.append({
                "scale": scale,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "target_width": self.detector_input_size[1],
                "target_height": self.detector_input_size[0],
            })

        batch_tensor = np.stack(batch, axis=0)
        batch_tensor = batch_tensor.transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
        return batch_tensor, batch_meta

    @staticmethod
    def _map_bbox_to_original(bbox, preprocess_meta):
        if len(bbox) != 4:
            return []
        if not preprocess_meta:
            return [int(round(v)) for v in bbox]

        x1, y1, x2, y2 = [float(v) for v in bbox]
        scale = float(preprocess_meta["scale"])
        x_offset = float(preprocess_meta["x_offset"])
        y_offset = float(preprocess_meta["y_offset"])
        target_width = int(preprocess_meta["target_width"])
        target_height = int(preprocess_meta["target_height"])

        original_width = int(round(target_width / scale)) if scale > 0 else target_width
        original_height = int(round(target_height / scale)) if scale > 0 else target_height

        x1 = (x1 - x_offset) / scale
        y1 = (y1 - y_offset) / scale
        x2 = (x2 - x_offset) / scale
        y2 = (y2 - y_offset) / scale

        x1 = max(0, min(original_width - 1, int(round(x1))))
        y1 = max(0, min(original_height - 1, int(round(y1))))
        x2 = max(0, min(original_width, int(round(x2))))
        y2 = max(0, min(original_height, int(round(y2))))
        return [x1, y1, x2, y2]

    @staticmethod
    def _expand_bbox(bbox, img_w, img_h, margin_ratio_x=0.08, margin_ratio_y=0.10):
        if len(bbox) != 4:
            return []
        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        margin_x = int(round(box_w * margin_ratio_x))
        margin_y = int(round(box_h * margin_ratio_y))
        return [
            max(0, x1 - margin_x),
            max(0, y1 - margin_y),
            min(img_w, x2 + margin_x),
            min(img_h, y2 + margin_y),
        ]

    @staticmethod
    def _crop_original_plate(image_bgr, bbox):
        if len(bbox) != 4:
            return np.empty((0, 0, 3), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return image_bgr[y1:y2, x1:x2]

    def _deduplicate_plates(self, plates):
        deduped = []
        for plate in plates:
            is_duplicate = False
            for kept in deduped:
                if self._bbox_iou(plate["bbox"], kept["bbox"]) > 0.9:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped.append(plate)
        return deduped

    def _parse_rt_detr(self, raw_output, img_w, img_h, conf_threshold, preprocess_meta=None):
        """
        Parses the [1, 300, 19] RT-DETR output tensor.
        """
        bboxes_normalized = raw_output[0, :, 0:4]  # [cx, cy, w, h]
        class_scores = raw_output[0, :, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)
        valid_mask = confidences > conf_threshold

        valid_bboxes_normalized = bboxes_normalized[valid_mask]
        valid_class_ids = class_ids[valid_mask]
        valid_confidences = confidences[valid_mask]

        results = []

        if preprocess_meta:
            scale = preprocess_meta["scale"]
            x_offset = preprocess_meta["x_offset"]
            y_offset = preprocess_meta["y_offset"]
            target_w = preprocess_meta["target_width"]
            target_h = preprocess_meta["target_height"]
        else:
            scale = 1.0
            x_offset = y_offset = 0.0
            target_w, target_h = img_w, img_h

        for bbox, score, class_id in zip(
                valid_bboxes_normalized, valid_confidences, valid_class_ids
        ):
            cx, cy, w, h = bbox
            cx_px = cx * target_w
            cy_px = cy * target_h
            w_px = w * target_w
            h_px = h * target_h

            x1_padded = cx_px - (w_px / 2)
            y1_padded = cy_px - (h_px / 2)
            x2_padded = cx_px + (w_px / 2)
            y2_padded = cy_px + (h_px / 2)

            x1 = max(0, int((x1_padded - x_offset) / scale))
            y1 = max(0, int((y1_padded - y_offset) / scale))
            x2 = min(img_w, int((x2_padded - x_offset) / scale))
            y2 = min(img_h, int((y2_padded - y_offset) / scale))

            if x2 > x1 and y2 > y1:
                results.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(score),
                    "class_id": int(class_id),
                    "class_name": self.vehicle_class_id_name_map.get(
                        int(class_id), "unknown"
                    ),
                })

        return results

    def _parse_plate_output_full_frame(self, raw_output, img_w, img_h, conf_threshold, preprocess_meta=None):
        """Parse plate detector output0 directly in shared 640x640 frame space."""
        if raw_output is None:
            return []

        if raw_output.ndim == 3 and raw_output.shape[0] == 1:
            raw_output = raw_output[0]
        if raw_output.ndim != 2 or raw_output.shape[1] < 5:
            return []

        # pb_utils.Logger.log_verbose(f"plate_output_shape={raw_output.shape}")
        boxes = raw_output[:, :4].astype(np.float32)
        if raw_output.shape[1] >= 6:
            confidences = raw_output[:, 4].astype(np.float32)
            class_ids = raw_output[:, 5].astype(np.int32)
        else:
            confidences = raw_output[:, 4].astype(np.float32)
            class_ids = zeros(raw_output.shape[0], dtype=np.int32)

        keep = confidences >= conf_threshold
        if not np.any(keep):
            # pb_utils.Logger.log_verbose(f"plate_detection_filtered_all conf_threshold={conf_threshold}")
            return []

        boxes = boxes[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]
        # pb_utils.Logger.log_verbose(f"plate_detection_count={len(boxes)}")

        results = []

        if preprocess_meta:
            scale = preprocess_meta["scale"]
            x_offset = preprocess_meta["x_offset"]
            y_offset = preprocess_meta["y_offset"]
            target_w = preprocess_meta["target_width"]
            target_h = preprocess_meta["target_height"]
        else:
            scale = 1.0
            x_offset = y_offset = 0.0
            target_w, target_h = img_w, img_h

        # Support multiple output bbox formats from plate models:
        # - xyxy absolute (pixel coords in detector input space)
        # - xyxy normalized (0..1)
        # - cxcywh normalized (0..1)
        max_abs_val = float(np.max(np.abs(boxes))) if boxes.size > 0 else 0.0
        is_normalized = max_abs_val <= 1.5
        if is_normalized:
            xyxy_like_ratio = float(np.mean((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])))
            normalized_format = "xyxy" if xyxy_like_ratio > 0.8 else "cxcywh"
        else:
            normalized_format = "xyxy_abs"

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if normalized_format == "xyxy_abs":
                x1_padded, y1_padded, x2_padded, y2_padded = [float(v) for v in box]
            elif normalized_format == "xyxy":
                x1_padded = float(box[0]) * target_w
                y1_padded = float(box[1]) * target_h
                x2_padded = float(box[2]) * target_w
                y2_padded = float(box[3]) * target_h
            else:
                cx, cy, w, h = [float(v) for v in box]
                cx_px = cx * target_w
                cy_px = cy * target_h
                w_px = w * target_w
                h_px = h * target_h
                x1_padded = cx_px - (w_px / 2)
                y1_padded = cy_px - (h_px / 2)
                x2_padded = cx_px + (w_px / 2)
                y2_padded = cy_px + (h_px / 2)

            # Remove padding and scale back to original image dimensions
            x1 = max(0, int((x1_padded - x_offset) / scale))
            y1 = max(0, int((y1_padded - y_offset) / scale))
            x2 = min(img_w, int((x2_padded - x_offset) / scale))
            y2 = min(img_h, int((y2_padded - y_offset) / scale))

            # pb_utils.Logger.log_verbose(f"plate_converted: bbox=[{x1},{y1},{x2},{y2}] valid={x2 > x1 and y2 > y1}")
            if x2 <= x1 or y2 <= y1:
                continue

            box_w = x2 - x1
            box_h = y2 - y1
            width_ratio = box_w / max(1, img_w)
            height_ratio = box_h / max(1, img_h)
            area_ratio = (box_w * box_h) / max(1, img_w * img_h)
            aspect_ratio = box_w / max(1, box_h)

            if width_ratio > 0.85 or height_ratio > 0.60 or area_ratio > 0.40:
                continue
            if width_ratio < 0.04 or height_ratio < 0.02:
                continue
            if aspect_ratio < 1.2 or aspect_ratio > 10.0:
                continue

            results.append(
                {"bbox": [x1, y1, x2, y2], "conf": float(conf), "class_id": int(class_id)}
            )

        # pb_utils.Logger.log_verbose(f"plate_parsed_count={len(results)}")
        return results

    @staticmethod
    def _bbox_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area / union)

    def _match_plates_to_vehicle(self, vehicle_bbox, plate_detections):
        """
        Match plates to a vehicle. Plates must be horizontally inside the vehicle's
        x-range (or very close), and vertically within the vehicle bbox or up to
        30% above/below it (to handle cases where the vehicle detector produces
        an undersized bbox that misses the bumper/grille area where plates sit).
        """
        matched = []
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vehicle_width = max(1, vx2 - vx1)
        vehicle_height = max(1, vy2 - vy1)

        for plate in plate_detections:
            px1, py1, px2, py2 = plate["bbox"]
            plate_w = max(1, px2 - px1)
            plate_h = max(1, py2 - py1)
            plate_center_x = (px1 + px2) / 2.0
            plate_center_y = (py1 + py2) / 2.0

            # Reject plates that are unreasonably large relative to the vehicle
            # (a real plate is much smaller than its parent vehicle).
            if plate_w > vehicle_width * 0.9 or plate_h > vehicle_height * 0.5:
                continue

            # Check 1: Plate center inside vehicle bbox
            if vx1 <= plate_center_x <= vx2 and vy1 <= plate_center_y <= vy2:
                matched.append(plate)
                continue

            # Check 2: IoU overlap
            if self._bbox_iou(vehicle_bbox, plate["bbox"]) > 0.05:
                matched.append(plate)
                continue

            # Check 3: Plate horizontally inside vehicle x-range and vertically
            # within 30% of vehicle height above/below (handles undersized
            # vehicle bboxes that miss the bumper/grille region).
            if vx1 <= plate_center_x <= vx2:
                vertical_margin = vehicle_height * 0.30
                if (vy1 - vertical_margin) <= plate_center_y <= (vy2 + vertical_margin):
                    matched.append(plate)
                    continue

        matched.sort(key=lambda item: item["conf"], reverse=True)
        return matched

    def _parse_ocr_output(self, raw_output):
        """
        Parse Paddle OCR output string like: 'AB12CD1234 (confidence: 0.87)'.
        """
        if raw_output is None or len(raw_output) == 0:
            return ("", 0.0)

        value = raw_output.flat[0]
        if isinstance(value, bytes):
            decoded = value.decode("utf-8")
        else:
            decoded = str(value)

        match = re.search(r"(.+?)\s*\(confidence:\s*([\d.]+)\)", decoded)
        if match:
            return (match.group(1).strip(), float(match.group(2)))
        if decoded.strip().lower() == "no text detected":
            return ("", 0.0)
        return (decoded.strip(), 0.0)

    def _parse_ocr_outputs(self, raw_output):
        if raw_output is None or len(raw_output) == 0:
            return []

        flattened = list(raw_output.flat)
        return [self._parse_ocr_output(np.array([value], dtype=object)) for value in flattened]

    def _prepare_plate_crop_for_ocr(self, plate_crop_bgr):
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            reason = "[ANPR_PIPELINE] [OCR] Empty plate crop received"
            pb_utils.Logger.log_warn(reason)
            return None, reason

        h, w = plate_crop_bgr.shape[:2]

        if h <= 0 or w <= 0:
            reason = "[ANPR_PIPELINE] [OCR] Invalid crop dimensions"
            pb_utils.Logger.log_warn(reason)
            return None, reason

        # ---------------- SIZE FILTER ----------------
        if not (self.ocr_min_width <= w <= self.ocr_max_width) or not (self.ocr_min_height <= h <= self.ocr_max_height):
            reason = f"[ANPR_PIPELINE] [OCR] Reject dimensions: {w}x{h}"
            if w < self.ocr_min_width:
                reason += f" (width too small: {w} < {self.ocr_min_width})"
            elif w > self.ocr_max_width:
                reason += f" (width too large: {w} > {self.ocr_max_width})"
            if h < self.ocr_min_height:
                reason += f" (height too small: {h} < {self.ocr_min_height})"
            elif h > self.ocr_max_height:
                reason += f" (height too large: {h} > {self.ocr_max_height})"
            pb_utils.Logger.log_warn(reason)
            return None, reason

        gray = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2GRAY)

        # ---------------- METRICS ----------------
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        # ---------------- FAST PATH (🔥 BIG WIN) ----------------
        # Widen thresholds to skip preprocessing for more images
        if 60 < brightness < 220 and lap_var > 20:
            return plate_crop_bgr, "[OCR] Good image → skipping preprocessing"

        # ---------------- PRIMARY FIXES (Mutually Exclusive) ----------------
        # We use if/elif so we don't destroy the image by applying 5 filters in a row.

        # ---------------- PRIMARY FIXES ----------------
        if brightness > 200:
            gray = cv2.LUT(gray, self.lut_darken)  # Use the > 1.0 Gamma
            gray = self.clahe_light.apply(gray)  # Restore contrast
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "[OCR] Overexposed → darkening"

        elif brightness < 60:
            gray = self.clahe_strong.apply(gray)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "[OCR] Underexposed → CLAHE strong"

        elif brightness > 180 and lap_var < 10:
            edges = cv2.Laplacian(gray, cv2.CV_8U)
            gray = cv2.addWeighted(gray, 0.85, edges, 0.15, 0)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "[OCR] IR reflection → edge boost"

        # ---------------- BLUR ----------------
        # Use ELIF here! If it was severely overexposed/underexposed and we fixed it,
        # don't immediately run a destructive sharpening kernel over it.
        elif lap_var < self.ocr_min_laplacian_variance:
            gray = cv2.filter2D(gray, -1, self.sharpen_kernel)
            if lap_var < 10:
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "[OCR] Blur detected → sharpening"
        else:
            # If it missed all the extreme lighting/blur checks but missed the fast path, apply light CLAHE
            gray = self.clahe_light.apply(gray)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "[OCR] Light CLAHE applied"

    def _normalize_plate_crop_for_ocr(self, plate_crop_bgr):
        return plate_crop_bgr

    def _pad_and_batch_ocr_inputs(self, crops):
        if not crops:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        max_h = max(crop.shape[0] for crop in crops)
        max_w = max(crop.shape[1] for crop in crops)
        resized_crops = []
        for crop in crops:
            # Resize to common size instead of black padding to preserve OCR quality
            if crop.shape[0] != max_h or crop.shape[1] != max_w:
                resized = cv2.resize(crop, (max_w, max_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized = crop
            resized_crops.append(resized)
        return np.stack(resized_crops, axis=0)

    # ----------------------------------------------------------
    # HELPERS FOR DEBUG ONLY DONT ENABLE IN PRODUCTION
    # ----------------------------------------------------------

    # ⚠️ Important: Don't log full string (too large)
    def _image_to_base64(self, image: np.ndarray, req_idx=None, img_idx=None):
        try:
            # Encode image to JPEG
            success, buffer = cv2.imencode(".jpg", image)
            if not success:
                pb_utils.Logger.log_error(
                    f"[ANPR_PIPELINE] [OCR][REQ-{req_idx}][IMG-{img_idx}] Failed to encode image"
                )
                return None

            # Convert to base64
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            pb_utils.Logger.log_warn(
                f"[ANPR_PIPELINE] [OCR][REQ-{req_idx}][IMG-{img_idx}] Base64 preview: {img_base64}"
            )

            return img_base64

        except Exception as e:
            pb_utils.Logger.log_error(
                f"[ANPR_PIPELINE] [OCR][REQ-{req_idx}][IMG-{img_idx}] Base64 conversion failed: {str(e)}"
            )
            return None

    def finalize(self):
        pass
```

### anpr_pipeline/config.pbtxt

```protobuf
name: "anpr_pipeline"
backend: "python"
max_batch_size: 10

input [
  {
    name: "image"
    data_type: TYPE_UINT8
    dims: [ -1, -1, 3 ]
  }
]

output [
  {
    name: "anpr_results"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]

dynamic_batching {
  preferred_batch_size: [2,4,6,8,10]
  max_queue_delay_microseconds: 10000
}

instance_group [
  {
    kind: KIND_CPU
    count: 1
    # gpus: [0]
  }
]


# Ensure the BLS script doesn't time out while waiting on the sub-models
model_transaction_policy {
  decoupled: False
}
```

---

## paddle_ocr_gpu_batched

### paddle_ocr_gpu_batched/model.py

```python
import numpy as np
import cv2
import os
import time
from statistics import mean
import triton_python_backend_utils as pb_utils

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.1"

# Local model paths to avoid downloading from internet
LOCAL_MODEL_DIR = "/models/paddle_ocr_gpu_batched/official_models"

MIN_OCR_SIZE = 128  # Upscale small plates for better OCR detection

# ----------------------------------------------------------
# UPDATED FOR PADDLEOCR 3.2 (PaddleX API)
# ----------------------------------------------------------
OCR_CONFIG = {
    # Lighter version
    # "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
    # "text_detection_model_name": "PP-OCRv5_mobile_det",

    # Heavier version (mixed: v5 detection + v5 recognition)
    # Server models are larger and more accurate than mobile versions
    "text_recognition_model_name": "PP-OCRv5_server_rec",
    "text_detection_model_name": "PP-OCRv5_server_det",

    # Text Detection Parameters
    # Since your input is already a cropped plate, the detector
    # doesn't need a massive 960x960 canvas. 320 is perfect.
    "text_det_limit_side_len": 320,  # Max dimension for detection input (larger = slower but more accurate)
    "text_det_box_thresh": 0.50,  # Minimum confidence for detected boxes (higher = faster, fewer false positives)
    "text_det_thresh": 0.30,  # Detection threshold for text regions (higher = faster, may miss faint text)
    "text_det_unclip_ratio": 1.60,  # Ratio to expand detected boxes (lower = tighter boxes, faster post-processing)
    "text_rec_score_thresh": 0.55,  # Minimum confidence for recognized text (higher = fewer false recognitions)

    # --- Preprocessing Module Disabling ---
    # These modules are disabled for ANPR to improve speed:
    # - License plates are already properly oriented (no rotation needed)
    # - License plates are flat surfaces (no perspective unwarping needed)
    # - License plate text is typically horizontal (no orientation classification needed)
    "use_doc_orientation_classify": False,  # Disable document orientation model
    "use_doc_unwarping": False,  # Disable document unwarping model
    "use_textline_orientation": False,  # Disable textline orientation model
}

class TritonPythonModel:

    def initialize(self, args):
        pb_utils.Logger.log_info("[OCR] Initializing PaddleOCR 3.2 (PaddleX) model")

        import paddle
        from paddleocr import PaddleOCR

        self._paddle = paddle
        self._device = self._select_device()
        self.log_border = "-" * 100

        # Initialize using the new PaddleX Wrapper parameters with local model paths
        det_model_path = os.path.join(
            LOCAL_MODEL_DIR, OCR_CONFIG["text_detection_model_name"]
        )
        rec_model_path = os.path.join(
            LOCAL_MODEL_DIR, OCR_CONFIG["text_recognition_model_name"]
        )

        pb_utils.Logger.log_info(f"[OCR] Using local detection model: {det_model_path}")
        pb_utils.Logger.log_info(
            f"[OCR] Using local recognition model: {rec_model_path}"
        )

        self.reader = PaddleOCR(
            det_model_dir=det_model_path,
            rec_model_dir=rec_model_path,
            text_det_limit_side_len=OCR_CONFIG["text_det_limit_side_len"],
            text_det_box_thresh=OCR_CONFIG["text_det_box_thresh"],
            text_det_thresh=OCR_CONFIG["text_det_thresh"],
            text_det_unclip_ratio=OCR_CONFIG["text_det_unclip_ratio"],
            text_rec_score_thresh=OCR_CONFIG["text_rec_score_thresh"],
            use_doc_orientation_classify=OCR_CONFIG["use_doc_orientation_classify"],
            use_doc_unwarping=OCR_CONFIG["use_doc_unwarping"],
            use_textline_orientation=OCR_CONFIG["use_textline_orientation"],
        )

        pb_utils.Logger.log_info(f"[OCR] Initialized on device: {self._device}")

    # ----------------------------------------------------------
    # MAIN EXECUTION
    # ----------------------------------------------------------
    def execute(self, requests):
        start_time = time.perf_counter()

        all_images = []
        request_map = []
        image_shapes = []
        start_msg = f"OCR EXEC START | total_images={len(requests)}"
        pb_utils.Logger.log_info(f"{self.log_border}")
        pb_utils.Logger.log_info(f"{start_msg}")
        pb_utils.Logger.log_info(f"{self.log_border}")

        # -------------------------------
        # 1. COLLECT INPUTS
        # -------------------------------
        for req_idx, request in enumerate(requests):
            try:
                in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT1")
                if in_tensor is None: continue

                arr = in_tensor.as_numpy()
                if arr is None or arr.size == 0: continue

                # batch handling
                if arr.ndim == 4:
                    images = list(arr)
                elif arr.ndim == 3:
                    images = [arr]
                else:
                    raise ValueError(f"Invalid shape: {arr.shape}")

                for i, img in enumerate(images):
                    if img is None or img.size == 0: continue
                    image_shapes.append((img.shape, img.dtype))
                    all_images.append(img)
                    request_map.append(req_idx)

            except Exception as e:
                pb_utils.Logger.log_error(f"[OCR][REQ-{req_idx}] Parsing failed: {str(e)}")

        total_images = len(all_images)
        pb_utils.Logger.log_info(
            f"[OCR] Aggregated batch → total_images={total_images}"
        )

        # -------------------------------
        # 2. EMPTY CASE
        # -------------------------------
        if total_images == 0:
            return self._empty_responses(requests)

        # -------------------------------
        # 3. PREPROCESS
        # -------------------------------
        t0 = time.perf_counter()
        processed_images = []
        for idx, img in enumerate(all_images):
            # Upscale if too small (Optional: helps mobile models read tiny crops)
            img = self._resize_if_small(img, idx)
            processed_images.append(img)

        preprocess_time = (time.perf_counter() - t0) * 1000

        # -------------------------------
        # 4. OCR (SINGLE GPU CALL)
        # -------------------------------
        pb_utils.Logger.log_info(f"[OCR] Running inference on {len(processed_images)} images")
        t1 = time.perf_counter()

        try:
            # Use predict() instead of deprecated ocr()
            # Since we are passing a list of images, it returns a list of results
            ocr_results = self.reader.predict(processed_images)
        except Exception as e:
            pb_utils.Logger.log_error(f"[OCR] Batch inference failed: {str(e)}")
            ocr_results = []

        ocr_time = (time.perf_counter() - t1) * 1000

        # -------------------------------
        # 5. POSTPROCESS
        # -------------------------------
        outputs = []
        success = 0

        # Handle empty results case
        if not ocr_results:
            ocr_results = [{}] * len(processed_images)

        for idx, res in enumerate(ocr_results):
            try:
                # PaddleX TextRecognition pipeline returns 'rec_text' and 'rec_score' directly
                # (or inside a 'res' key depending on exact sub-pipeline output formatting)
                if isinstance(res, dict):
                    texts = res.get("rec_text", res.get("rec_texts", []))
                    scores = res.get("rec_score", res.get("rec_scores", []))
                else:
                    # If it's a PaddleX result object
                    texts = getattr(res, "rec_text", getattr(res, "rec_texts", []))
                    scores = getattr(res, "rec_score", getattr(res, "rec_scores", []))

                # Normalize to lists for iteration
                if not isinstance(texts, list): texts = [texts] if texts else []
                if not isinstance(scores, list): scores = [scores] if scores else []

                shape, dtype = image_shapes[idx] if idx < len(image_shapes) else (None, None)
                pb_utils.Logger.log_verbose(f"[OCR][IMG-{idx}] shape={shape} | texts={texts} scores={scores}")

                success += 1

                if texts and scores and texts[0]:
                    combined_text = "".join(str(t or "") for t in texts)
                    combined_score = float(mean(float(s) for s in scores if s is not None))
                    result_str = f"{combined_text} (confidence: {combined_score:.2f})"
                    outputs.append(result_str)
                else:
                    outputs.append("")

            except Exception as e:
                pb_utils.Logger.log_error(f"[OCR][IMG-{idx}] Postprocess failed: {str(e)}")
                outputs.append("Postprocess error")

        # -------------------------------
        # 6. MAP BACK & BUILD RESPONSES
        # -------------------------------
        per_request_outputs = [[] for _ in requests]
        for out, req_idx in zip(outputs, request_map):
            per_request_outputs[req_idx].append(out)

        responses = []
        for req_idx, out_list in enumerate(per_request_outputs):
            if not out_list:
                pb_utils.Logger.log_warn(
                    f"[OCR][REQ-{req_idx}] No output generated"
                )
                out_list = ["No input"]
            out_arr = np.array(out_list, dtype=object).reshape(-1, 1)
            responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("OUTPUT0", out_arr)]))
        # -------------------------------
        # 8. FINAL METRICS
        # -------------------------------
        total_time = (time.perf_counter() - start_time) * 1000
        pb_utils.Logger.log_verbose(
            f"[OCR] DONE | total={total_time:.2f}ms | preprocess={preprocess_time:.2f}ms | "
            f"ocr={ocr_time:.2f}ms | images={total_images} | success={success} | "
            f"per_image={total_time / max(1, total_images):.2f}ms"
        )
        end_msg = f"OCR EXEC END | Total: {total_time:.2f}ms"
        pb_utils.Logger.log_info(f"{self.log_border}")
        pb_utils.Logger.log_info(f"{end_msg}")
        pb_utils.Logger.log_info(f"{self.log_border}")

        return responses

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------
    def _resize_if_small(self, image, idx):
        h, w = image.shape[:2]
        if h < MIN_OCR_SIZE or w < MIN_OCR_SIZE:
            scale = max(MIN_OCR_SIZE / h, MIN_OCR_SIZE / w)
            new_size = (int(w * scale), int(h * scale))
            pb_utils.Logger.log_verbose(
                f"[OCR][IMG-{idx}] Upscaling {image.shape} → {new_size}"
            )
            image = cv2.resize(image, new_size)
        return image

    def _empty_responses(self, requests):
        responses = []
        for _ in requests:
            out = np.array(["No input"], dtype=object).reshape(-1, 1)
            responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("OUTPUT0", out)]))
        return responses

    def _select_device(self):
        if self._paddle.device.is_compiled_with_cuda():
            try:
                self._paddle.set_device("gpu:0")
                return "gpu:0"
            except Exception:
                pass
        self._paddle.set_device("cpu")
        return "cpu"
```

### paddle_ocr_gpu_batched/config.pbtxt

```protobuf
name: "paddle_ocr_gpu_batched"
backend: "python"
max_batch_size: 96

dynamic_batching {
  preferred_batch_size: [ 16, 32, 64, 96 ]
  max_queue_delay_microseconds: 10000
}

input [
  {
    name: "INPUT1"
    data_type: TYPE_UINT8
    dims: [ -1, -1, 3 ]
    optional: true
  }
]
output [
  {
    name: "OUTPUT0"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

instance_group [
  {
    kind: KIND_GPU
    count: 1
    gpus: [0]
  }
]
```

---

## docker-compose.yml

```yaml
services:
  triton-server-v2:
    image: triton-server:v2.0.0
    container_name: triton-server-v2
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    environment:
      TRITON_HTTP_PORT: "8000"
      TRITON_GRPC_PORT: "8001"
      TRITON_METRICS_PORT: "8002"
      TRITON_LOG_VERBOSE: "0"
    volumes:
      - ./model_repository:/models
    shm_size: '2g'
    gpus:
      - device: 0
        capabilities: [compute, utility]
    # Triton server command with optimized settings
    command: >
      tritonserver
      --model-repository=/models
      --http-port=8000
      --grpc-port=8001
      --metrics-port=8002
      --allow-metrics=true
      --allow-gpu-metrics=true
      --log-verbose=0
      --log-info=true
      --log-warning=true
      --log-error=true
      --exit-on-error=false
      --model-control-mode=none
      --grpc-infer-allocation-pool-size=16
      --grpc-max-response-pool-size=16
      --cuda-memory-pool-byte-size=0:2147483648
      --pinned-memory-pool-byte-size=268435456
      --metrics-interval-ms=2000
```

---

**End of Document**
