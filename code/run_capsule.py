import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
import argparse
import sys
import numpy as np
from pathlib import Path
import json
import logging

# ABSOLUTE PROBE JSON PATH FOR NIDQ
PROBE_ABS = "/home/tsw454/ephys-spike-sorting/pipeline/hms_cluster/nidq_probe.json"

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.core.core_tools import SIJsonEncoder
import probeinterface as pi

try:
    from aind_log_utils import log
    HAVE_AIND_LOG_UTILS = True
except ImportError:
    HAVE_AIND_LOG_UTILS = False

# timestamp tolerance constants
ACCEPTED_NEGATIVE_DEVIATION_MS = 0.2
MAX_NUM_NEGATIVE_TIMESTAMPS = 10
ABS_MAX_TIMESTAMPS_DEVIATION_MS = 2
MAX_TIMESTAMPS_DEVIATION_MS = 1

data_folder = Path("../data")
results_folder = Path("../results")

# -------------------- ARGPARSE --------------------
parser = argparse.ArgumentParser(description="Dispatch jobs for AIND ephys pipeline")

split_segment_group = parser.add_mutually_exclusive_group()
split_segment_help = "Whether to concatenate or split recording segments or not. Default: split segments"
split_segment_group.add_argument("--no-split-segments", action="store_true", help=split_segment_help)
split_segment_group.add_argument("static_split_segments", nargs="?", help=split_segment_help)

split_group = parser.add_mutually_exclusive_group()
split_help = "Whether to process different groups separately. Default: split groups"
split_group.add_argument("--no-split-groups", action="store_true", help=split_help)
split_group.add_argument("static_split_groups", nargs="?", help=split_help)

debug_group = parser.add_mutually_exclusive_group()
debug_help = "Whether to run in DEBUG mode. Default: False"
debug_group.add_argument("--debug", action="store_true", help=debug_help)
debug_group.add_argument("static_debug", nargs="?", help=debug_help)

debug_duration_group = parser.add_mutually_exclusive_group()
debug_duration_help = "Duration of clipped recording in debug mode. Default: 30 seconds"
debug_duration_group.add_argument("--debug-duration", default=30, help=debug_duration_help)
debug_duration_group.add_argument("static_debug_duration", nargs="?", default=None, help=debug_duration_help)

timestamps_skip_group = parser.add_mutually_exclusive_group()
timestamps_skip_help = "Skip timestamps check"
timestamps_skip_group.add_argument("--skip-timestamps-check", action="store_true", help=timestamps_skip_help)
timestamps_skip_group.add_argument("static_skip_timestamps_check", nargs="?", help=timestamps_skip_help)

input_group = parser.add_mutually_exclusive_group()
input_help = "Which 'loader' to use (spikeglx | openephys | nwb | spikeinterface | aind)"
input_group.add_argument("--input", default=None, help=input_help, choices=["aind", "spikeglx", "openephys", "nwb", "spikeinterface"])
input_group.add_argument("static_input", nargs="?", help=input_help)

multi_session_group = parser.add_mutually_exclusive_group()
multi_session_help = "Whether the data folder includes multiple sessions or not. Default: False"
multi_session_group.add_argument("--multi-session", action="store_true", help=multi_session_help)
multi_session_group.add_argument("static_multi_session", nargs="?", help=multi_session_help)

min_recording_duration = parser.add_mutually_exclusive_group()
min_recording_duration_help = "Skip recordings shorter than this value (seconds). -1 disables."
min_recording_duration.add_argument("--min-recording-duration", default="-1", help=min_recording_duration_help)
min_recording_duration.add_argument("static_min_recording_duration", nargs="?", default=None, help=min_recording_duration_help)

spikeinterface_info_group = parser.add_mutually_exclusive_group()
spikeinterface_info_help = """
SpikeInterface loader info JSON path or string. Only needed if --input spikeinterface.
"""
spikeinterface_info_group.add_argument("--spikeinterface-info", default=None, help=spikeinterface_info_help)
spikeinterface_info_group.add_argument("static_spikeinterface_info", nargs="?", default=None, help=spikeinterface_info_help)

parser.add_argument("--params", default=None, help="Path to a params JSON or a JSON string. Overrides all other args.")

if __name__ == "__main__":
    args = parser.parse_args()

    # -------------------- PARAMS MERGE --------------------
    PARAMS = args.params
    if PARAMS is not None:
        try:
            params = json.loads(PARAMS)
        except json.JSONDecodeError:
            if Path(PARAMS).is_file():
                params = json.loads(Path(PARAMS).read_text())
            else:
                raise ValueError(f"Invalid parameters: {PARAMS} is not JSON or a file path")

        SPLIT_SEGMENTS = params.get("split_segments", False)
        SPLIT_GROUPS = params.get("split_groups", True)
        DEBUG = params.get("debug", False)
        DEBUG_DURATION = float(params.get("debug_duration"))
        SKIP_TIMESTAMPS_CHECK = params.get("skip_timestamps_check", False)
        MULTI_SESSION = params.get("multi_session", False)
        INPUT = params.get("input"); assert INPUT is not None, "Input type is required"
        if INPUT == "spikeinterface":
            spikeinterface_info = params.get("spikeinterface_info")
            assert spikeinterface_info is not None, "SpikeInterface info is required with spikeinterface loader"
        MIN_RECORDING_DURATION = params.get("min_recording_duration", -1)
    else:
        SPLIT_SEGMENTS = (args.static_split_segments.lower() == "true" if args.static_split_segments else not args.no_split_segments)
        SPLIT_GROUPS = (args.static_split_groups.lower() == "true" if args.static_split_groups else not args.no_split_groups)
        DEBUG = (args.static_debug.lower() == "true" if args.static_debug else args.debug)
        DEBUG_DURATION = float(args.static_debug_duration or args.debug_duration)
        SKIP_TIMESTAMPS_CHECK = (args.static_skip_timestamps_check.lower() == "true" if args.static_skip_timestamps_check else args.skip_timestamps_check)
        MULTI_SESSION = (args.static_multi_session.lower() == "true" if args.static_multi_session else args.multi_session)
        INPUT = args.static_input or args.input
        if INPUT == "spikeinterface":
            spikeinterface_info = args.static_spikeinterface_info or args.spikeinterface_info
            assert spikeinterface_info is not None, "SpikeInterface info is required with spikeinterface loader"
        MIN_RECORDING_DURATION = float(args.static_min_recording_duration or args.min_recording_duration)

    # -------------------- LOGGING --------------------
    aind_log_setup = False
    ecephys_session_folders = None
    if INPUT == "aind":
        ecephys_session_folders = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower() or "behavior" in p.name.lower()]
        if len(ecephys_session_folders) == 0:
            raise Exception("No valid ecephys sessions found.")
        elif len(ecephys_session_folders) > 1 and not MULTI_SESSION:
            raise Exception("Multiple ecephys sessions found. Please only add one at a time")

    if HAVE_AIND_LOG_UTILS and ecephys_session_folders is not None:
        ecephys_session_folder = ecephys_session_folders[0]
        subject_json = ecephys_session_folder / "subject.json"
        subject_id = "undefined"
        if subject_json.is_file():
            subject_id = json.load(open(subject_json, "r"))["subject_id"]
        data_description_json = ecephys_session_folder / "data_description.json"
        session_name = "undefined"
        if data_description_json.is_file():
            session_name = json.load(open(data_description_json, "r"))["name"]
        log.setup_logging("Job Dispatch Ecephys", subject_id=subject_id, asset_name=session_name)
        aind_log_setup = True
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    logging.info("Running job dispatcher with parameters:")
    logging.info(f"\tSPLIT SEGMENTS: {SPLIT_SEGMENTS}")
    logging.info(f"\tSPLIT GROUPS: {SPLIT_GROUPS}")
    logging.info(f"\tDEBUG: {DEBUG}")
    logging.info(f"\tDEBUG DURATION: {DEBUG_DURATION}")
    logging.info(f"\tSKIP TIMESTAMPS CHECK: {SKIP_TIMESTAMPS_CHECK}")
    logging.info(f"\tMULTI SESSION: {MULTI_SESSION}")
    logging.info(f"\tINPUT: {INPUT}")
    logging.info(f"\tMIN_RECORDING_DURATION: {MIN_RECORDING_DURATION}")

    logging.info(f"Parsing {INPUT} input folder")
    recording_dict = {}
    include_annotations = False

    # -------------------- INPUT MODES --------------------
    if INPUT == "aind":
        # (unchanged from your version) ...
        # [omitted for brevity — identical to your original block]
        pass

    elif INPUT == "spikeglx":
        spikeglx_folders = [p for p in data_folder.iterdir() if p.is_dir()]
        if len(spikeglx_folders) == 0:
            raise Exception("No valid SpikeGLX folder found.")
        elif len(spikeglx_folders) > 1 and not MULTI_SESSION:
            raise Exception("Multiple SpikeGLX sessions found. Please only add one at a time")

        for spikeglx_folder in spikeglx_folders:
            session_name = spikeglx_folder.name
            stream_names, stream_ids = se.get_neo_streams("spikeglx", spikeglx_folder)
            num_blocks = 1
            block_index = 0

            logging.info(f"\tSession name: {session_name}")
            logging.info(f"\tNum. streams: {len(stream_names)}")
            for stream_name in stream_names:
                # keep AP and NIDQ; drop LF and SYNC
                if "lf" not in stream_name and "SYNC" not in stream_name:
                    recording = se.read_spikeglx(spikeglx_folder, stream_name=stream_name)
                    recording_name = f"block{block_index}_{stream_name}_recording"
                    recording_dict[(session_name, recording_name)] = {
                        "input_folder": spikeglx_folder,
                        "raw": recording,
                        "stream_name": stream_name,  # save for later (to tag nidq)
                    }

                    if "ap" in stream_name:
                        stream_name_lf = stream_name.replace("ap", "lf")
                        try:
                            recording_lf = se.read_spikeglx(spikeglx_folder, stream_name=stream_name_lf)
                            recording_dict[(session_name, recording_name)]["lfp"] = recording_lf
                        except Exception:
                            logging.info(f"\t\tNo LFP stream found for {stream_name}")

    elif INPUT == "openephys":
        # (unchanged from your version) ...
        pass

    elif INPUT == "nwb":
        # (unchanged from your version) ...
        pass

    elif INPUT == "spikeinterface":
        # (unchanged from your version) ...
        pass

    # -------------------- BUILD JOB JSONS --------------------
    job_dict_list = []
    logging.info("Recording(s) to be processed in parallel:")
    for session_recording_name in recording_dict:
        session_name, recording_name = session_recording_name
        input_folder = recording_dict[session_recording_name].get("input_folder")
        recording = recording_dict[session_recording_name]["raw"]
        recording_lfp = recording_dict[session_recording_name].get("lfp", None)
        orig_stream_name = recording_dict[session_recording_name].get("stream_name", None)

        if MIN_RECORDING_DURATION != -1:
            duration = recording.get_total_duration()
            if duration < MIN_RECORDING_DURATION:
                logging.info(f"\tSkipping {session_name}-{recording_name} (duration {np.round(duration, 2)}s)")
                continue

        HAS_LFP = recording_lfp is not None
        if not SPLIT_SEGMENTS:
            recordings = [recording]
            recordings_lfp = [recording_lfp] if HAS_LFP else None
        else:
            recordings = si.split_recording(recording)
            recordings_lfp = si.split_recording(recording_lfp) if HAS_LFP else None

        for recording_index, rec_one in enumerate(recordings):
            rec_name_seg = f"{recording_name}{recording_index + 1}" if SPLIT_SEGMENTS else recording_name
            if HAS_LFP:
                recording_lfp_one = recordings_lfp[recording_index]

            # timestamps sanity
            skip_times = False
            if not SKIP_TIMESTAMPS_CHECK:
                for segment_index in range(rec_one.get_num_segments()):
                    times = rec_one.get_times(segment_index=segment_index)
                    diffs_ms = np.diff(times) * 1000
                    if np.sum(diffs_ms < -ACCEPTED_NEGATIVE_DEVIATION_MS) > MAX_NUM_NEGATIVE_TIMESTAMPS:
                        skip_times = True; break
                    if np.max(np.abs(diffs_ms)) > ABS_MAX_TIMESTAMPS_DEVIATION_MS:
                        skip_times = True; break
            if skip_times:
                rec_one.reset_times()

            # debug clipping
            if DEBUG:
                parts = []
                for segment_index in range(rec_one.get_num_segments()):
                    rseg = si.split_recording(rec_one)[segment_index]
                    rseg = rseg.frame_slice(0, min(int(DEBUG_DURATION * rec_one.sampling_frequency), rseg.get_num_samples()))
                    parts.append(rseg)
                rec_one = si.append_recordings(parts)
                if HAS_LFP:
                    parts_lf = []
                    for segment_index in range(recording_lfp_one.get_num_segments()):
                        rseg = si.split_recording(recording_lfp_one)[segment_index]
                        rseg = rseg.frame_slice(0, min(int(DEBUG_DURATION * recording_lfp_one.sampling_frequency), rseg.get_num_samples()))
                        parts_lf.append(rseg)
                    recording_lfp_one = si.append_recordings(parts_lf)

            duration = np.round(rec_one.get_total_duration(), 2)

            # split by groups if any
            def _emit_job(_rec, _name_suffix, _lfp=None):
                job = dict(
                    session_name=session_name,
                    recording_name=str(_name_suffix),
                    recording_dict=_rec.to_dict(recursive=True, include_annotations=False, relative_to=data_folder),
                    skip_times=skip_times,
                    duration=duration,
                    input_folder=input_folder,
                    debug=DEBUG,
                )
                # >>> attach absolute probe path for NIDQ
                if (INPUT == "spikeglx") and (isinstance(orig_stream_name, str)) and (orig_stream_name.lower() == "nidq"):
                    job["probe_paths"] = PROBE_ABS  # absolute path requested
                # <<<

                if _lfp is not None:
                    job["recording_lfp_dict"] = _lfp.to_dict(recursive=True, relative_to=data_folder)
                return job

            if SPLIT_GROUPS and len(np.unique(rec_one.get_channel_groups())) > 1:
                for group_name, rec_group in rec_one.split_by("group").items():
                    name_group = f"{rec_name_seg}_group{group_name}"
                    job = _emit_job(rec_group, name_group, _lfp=(recording_lfp_one.split_by("group")[group_name] if HAS_LFP else None))
                    logging.info(f"\t{name_group}\n\t\tDuration {duration} s - Num. channels: {rec_group.get_num_channels()}")
                    job_dict_list.append(job)
            else:
                job = _emit_job(rec_one, rec_name_seg, _lfp=(recording_lfp_one if HAS_LFP else None))
                logging.info(f"\t{rec_name_seg}\n\t\tDuration: {duration} s - Num. channels: {rec_one.get_num_channels()}")
                job_dict_list.append(job)

    if not results_folder.is_dir():
        results_folder.mkdir(parents=True)

    if MULTI_SESSION:
        logging.info("Adding session name to recording name")

    for i, job_dict in enumerate(job_dict_list):
        job_dict["multi_input"] = MULTI_SESSION
        if MULTI_SESSION:
            job_dict["recording_name"] = f"{job_dict['session_name']}__{job_dict['recording_name']}"
        with open(results_folder / f"job_{i}.json", "w") as f:
            json.dump(job_dict, f, indent=4, cls=SIJsonEncoder)

    logging.info(f"Generated {len(job_dict_list)} job config files")
