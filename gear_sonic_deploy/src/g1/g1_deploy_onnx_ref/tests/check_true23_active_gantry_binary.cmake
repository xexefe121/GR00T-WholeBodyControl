if(NOT DEFINED BINARY OR NOT EXISTS "${BINARY}")
  message(FATAL_ERROR "true23 active gantry binary does not exist: ${BINARY}")
endif()
if(NOT DEFINED SOURCE OR NOT EXISTS "${SOURCE}")
  message(FATAL_ERROR "true23 active gantry source does not exist: ${SOURCE}")
endif()

find_program(STRINGS_EXECUTABLE strings)
if(NOT STRINGS_EXECUTABLE)
  message(FATAL_ERROR "strings is required for active surface audit")
endif()
execute_process(
  COMMAND "${STRINGS_EXECUTABLE}" -a "${BINARY}"
  RESULT_VARIABLE STRINGS_RESULT
  OUTPUT_VARIABLE BINARY_SURFACE
  ERROR_VARIABLE STRINGS_ERROR)
if(NOT STRINGS_RESULT EQUAL 0)
  message(FATAL_ERROR "strings failed: ${STRINGS_ERROR}")
endif()
file(READ "${SOURCE}" SOURCE_SURFACE)
set(COMBINED_SURFACE "${BINARY_SURFACE}\n${SOURCE_SURFACE}")

foreach(FORBIDDEN
    "disable-crc"
    "disableCrcCheck"
    "--mode control"
    "g1_deploy_onnx_ref.cpp"
    "SetFsmId(released.locomotion_fsm_id)"
    "publisher->Write(ToLowCmd(value.BuildDampingCommand()))"
    "kFaultDampingCycles"
    "fail-safe damping")
  string(FIND "${COMBINED_SURFACE}" "${FORBIDDEN}" FOUND_OFFSET)
  if(NOT FOUND_OFFSET EQUAL -1)
    message(FATAL_ERROR
      "active gantry target contains forbidden bypass/generic surface: ${FORBIDDEN}")
  endif()
endforeach()

foreach(RUNTIME_REQUIRED
    "I_CONFIRM_G1_TRUE23_STAGE1_GANTRY"
    "rt/lowstate"
    "rt/lowcmd"
    "deployment_ready"
    "active_motor_control_authorized"
    "gantry_authorized"
    "--validate-only"
    "--live-shadow-evidence"
    "--authorization-id"
    "--frozen-lora-policy"
    "--evidence"
    "--post-arm-duration-seconds"
    "duplicate option rejected"
    "operator authorization-id does not match active promotion"
    "[PASS] exact True23 "
    " gantry promotion and "
    "ValidateLiveShadowEvidenceJsonl"
    "promoted_shadow"
    "session_complete"
    "record count does not exactly match requested frames"
    "live shadow evidence is not fresh"
    "signal stopped controller before motion-mode release"
    "signal stopped controller before read-only policy prewarm"
    "pre-arm posture hold was not ready with zero LowCmd writes"
    "writer completion time is invalid"
    "execution evidence path identity changed during session"
    "session_no_actuation"
    "[NO ACTUATION]"
    "first_policy_ready_for_arm"
    "reviewed_post_arm_duration_complete"
    "post_arm_elapsed_ns"
    "[PREWARM] Unitree motion mode retained; zero LowCmd writes"
    "waiting for first fresh 10-frame causal policy"
    "pre_arm_hold_prepared"
    "pre_arm_hold_gate_open"
    "normal_return_hold_started"
    "motion_mode_restored"
    "startup_damping_frames"
    "sampled_posture_hold"
    "g1_true23_causal_mujoco_promotion"
    "deployment_bytes_authorized"
    "executed_producer_archive_manifest_sha256"
    "decoder_output_semantics"
    "applied_safe_native_action"
    "safe_target_transform_sha256"
    "action_clip_value must be exactly 20"
    "decoder must have exactly 23 native outputs"
    "five advancing CRC-valid mode_machine==4"
    "g1_true23_causal_history_reference_terms"
    "ParseCausalPicoReferenceTerms"
    "BuildCausalEncoderInput"
    "causal artifact rejects future-command PICO fields")
  string(FIND "${BINARY_SURFACE}" "${RUNTIME_REQUIRED}" FOUND_OFFSET)
  if(FOUND_OFFSET EQUAL -1)
    message(FATAL_ERROR
      "active gantry binary is missing runtime gate: ${RUNTIME_REQUIRED}")
  endif()
endforeach()

foreach(SOURCE_REQUIRED
    "frame_index != *last_frame + 1U"
    "*last_source_monotonic_ns + active::kShadowControlPeriodNs"
    "NextNoCatchUpWriterDeadlineNs(completed_write_ns)"
    "first_policy_ready_for_arm.store(true, std::memory_order_release)"
    "stop_reason == \"reviewed_post_arm_duration_complete\""
    "monitor.WaitJoinCausalReference"
    "value.PreparePreArmHold(hold_prepare_ns)"
    "value.BeginNormalReturnHold(now_ns)"
    "value.BeginSoftwareFaultReturnHold(recovery_ns)"
    "RestoreMotionModeAfterNormalHold(released_motion_mode)"
    "active::IsPositiveGainRuntimeCommand(command)"
    "mode_handoff_interlock.Request()"
    "WaitForWriterQuiescence(mode_handoff_interlock)"
    "writer_quiesced_before_select"
    "publisher.reset()"
    "lowcmd_publisher_closed_before_select"
    "SelectMode(released.name) == 0"
    "outgoing non-positive-gain LowCmd rejected before DDS"
    "emergency_motion_mode_restored"
    "writer_emergency_mode_handoff"
    "software_transport_normal_return"
    "!motion_mode_released.load(std::memory_order_acquire)"
    "startup_damping_frames.load(std::memory_order_acquire) == 0"
    "g1_true23_frozen_lora_dance_gantry_active_promotion_v2"
    "g1_true23_frozen_lora_live_gantry_active_promotion_v1"
    "live::BuildCausalEncoderInput"
    "if (g_stop_requested != 0)")
  string(FIND "${SOURCE_SURFACE}" "${SOURCE_REQUIRED}" FOUND_OFFSET)
  if(FOUND_OFFSET EQUAL -1)
    message(FATAL_ERROR
      "active gantry source is missing structural gate: ${SOURCE_REQUIRED}")
  endif()
endforeach()

message(STATUS "true23 active gantry surface audit passed")
