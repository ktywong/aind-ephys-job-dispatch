import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
import argparse
import sys
import numpy as np
import warnings
from pathlib import Path
import json
import logging


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


# here we define some constants used for defining if timestamps are ok
# or should be skipped
ACCEPTED_NEGATIVE_DEVIATION_MS = 0.2  # we allow for small negative timestamps diff glitches
MAX_NUM_NEGATIVE_TIMESTAMPS = 10  # maximum number of negative timestamps allowed below the accepted deviation
ABS_MAX_TIMESTAMPS_DEVIATION_MS = 2  # absolute maximum deviation allowed for timestamps (also positive)

MAX_NUM_NEGATIVE_TIMESTAMPS = 10
MAX_TIMESTAMPS_DEVIATION_MS = 1


data_folder = Path("../data")
results_folder = Path("../results")


# Define argument parser
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
debug_duration_help = (
    "Duration of clipped recording in debug mode. Only used if debug is enabled. Default: 30 seconds"
)
debug_duration_group.add_argument("--debug-duration", default=30, help=debug_duration_help)
debug_duration_group.add_argument("static_debug_duration", nargs="?", default=None, help=debug_duration_help)

timestamps_skip_group = parser.add_mutually_exclusive_group()
timestamps_skip_help = "Skip timestamps check"
timestamps_skip_group.add_argument("--skip-timestamps-check", action="store_true", help=timestamps_skip_help)
timestamps_skip_group.add_argument(
    "static_skip_timestamps_check", nargs="?", help=timestamps_skip_help
)

input_group = parser.add_mutually_exclusive_group()
input_help = "Which 'loader' to use (spikeglx | openephys | nwb | spikeinterface | aind)"
input_group.add_argument("--input", default=None, help=input_help, choices=["aind", "spikeglx", "openephys", "nwb", "spikeinterface"])
input_group.add_argument("static_input", nargs="?", help=input_help)

multi_session_group = parser.add_mutually_exclusive_group()
multi_session_help = "Whether the data folder includes multiple sessions or not. Default: False"
multi_session_group.add_argument("--multi-session", action="store_true", help=multi_session_help)
multi_session_group.add_argument("static_multi_session", nargs="?", help=multi_session_help)

min_recording_duration = parser.add_mutually_exclusive_group()
min_recording_duration_help = (
    "If provided, skips recordings with duration less than this value. If -1 (default), no recordings are skipped"
)
min_recording_duration.add_argument("--min-recording-duration", default="-1", help=min_recording_duration_help)
min_recording_duration.add_argument("static_min_recording_duration", nargs="?", default=None, help=min_recording_duration_help)

spikeinterface_info_group = parser.add_mutually_exclusive_group()
spikeinterface_info_help = """
    A JSON path or string to specify how to parse the recording in spikeinterface, including: 
    - 1. reader_type (required): string with the reader type (e.g. 'plexon', 'neuralynx', etc.) .
    - 2. reader_kwargs (optional): dictionary (or list of dicts for multi-session) with the reader kwargs (e.g. {'folder': '/path/to/folder'}).
    - 3. keep_stream_substrings (optional): string or list of strings with the stream names to load (e.g. 'AP' or ['AP', 'LFP']).
    - 4. skip_stream_substrings (optional): string (or list of strings) with substrings used to skip streams (e.g. 'NIDQ' or ['USB', 'EVTS']).
    - 5. probe_paths (optional): string or dict with the probe paths to a ProbeInterface JSON file (e.g. '/path/to/probe.json'). 
                                 If a dict is provided, the key is the stream name and the value is the probe path.
                                 If multi-session, a list can be provided with the probe paths for each session.
    - 6. session_names (optional): string or list of strings with the session names (e.g. 'session1').
    If reader_kwargs is not provided, the reader will be created with default parameters. The probe_path is 
    required if the reader doesn't load the probe automatically.
"""
spikeinterface_info_group.add_argument(
    "--spikeinterface-info",
    default=None,
    help=spikeinterface_info_help,
)
spikeinterface_info_group.add_argument(
    "static_spikeinterface_info",
    nargs="?",
    default=None,
    help=spikeinterface_info_help,
)

parser.add_argument("--params", default=None, help="Path to the parameters file or JSON string. If given, it will override all other arguments.")

if __name__ == "__main__":
    args = parser.parse_args()

    # if params is given, override all other arguments
    PARAMS = args.params
    if PARAMS is not None:
        # try to parse the JSON string first to avoid file name too long error
        try:
            params = json.loads(PARAMS)
        except json.JSONDecodeError:
            if Path(PARAMS).is_file():
                with open(PARAMS, "r") as f:
                    params = json.load(f)
            else:
                raise ValueError(f"Invalid parameters: {PARAMS} is not a valid JSON string or file path")

        SPLIT_SEGMENTS = params.get("split_segments", False)
        SPLIT_GROUPS = params.get("split_groups", True)
        DEBUG = params.get("debug", False)
        DEBUG_DURATION = float(params.get("debug_duration"))
        SKIP_TIMESTAMPS_CHECK = params.get("skip_timestamps_check", False)
        MULTI_SESSION = params.get("multi_session", False)
        INPUT = params.get("input")
        assert INPUT is not None, "Input type is required"
        if INPUT == "spikeinterface":
            spikeinterface_info = params.get("spikeinterface_info")
            assert spikeinterface_info is not None, "SpikeInterface info is required when using the spikeinterface loader"
        MULTI_SESSION = params.get("multi_session", False)
        MIN_RECORDING_DURATION = params.get("min_recording_duration", -1)
    else:
        # if params is not given, use the arguments
        SPLIT_SEGMENTS = (
            args.static_split_segments.lower() == "true" if args.static_split_segments
            else not args.no_split_segments
        )
        SPLIT_GROUPS = (
            args.static_split_groups.lower() == "true" if args.static_split_groups
            else not args.no_split_groups
        )
        DEBUG = (
            args.static_debug.lower() == "true" if args.static_debug
            else args.debug
        )
        DEBUG_DURATION = float(args.static_debug_duration or args.debug_duration)
        SKIP_TIMESTAMPS_CHECK = (
            args.static_skip_timestamps_check.lower() == "true" if args.static_skip_timestamps_check
            else args.skip_timestamps_check
        )
        MULTI_SESSION = (
            args.static_multi_session.lower() == "true" if args.static_multi_session
            else args.multi_session
        )
        INPUT = args.static_input or args.input
        if INPUT == "spikeinterface":
            spikeinterface_info = args.static_spikeinterface_info or args.spikeinterface_info
            assert spikeinterface_info is not None, "SpikeInterface info is required when using the spikeinterface loader"
        MIN_RECORDING_DURATION = float(args.static_min_recording_duration or args.min_recording_duration)

    # setup AIND logging before any other logging call
    aind_log_setup = False

    ecephys_session_folders = None
    if INPUT == "aind":
        ecephys_session_folders = [
            p for p in data_folder.iterdir() if "ecephys" in p.name.lower() or "behavior" in p.name.lower()
        ]
        if len(ecephys_session_folders) == 0:
            raise Exception("No valid ecephys sessions found.")
        elif len(ecephys_session_folders) > 1:
            if not MULTI_SESSION:
                raise Exception("Multiple ecephys sessions found in the data folder. Please only add one at a time")


    if HAVE_AIND_LOG_UTILS and ecephys_session_folders is not None:
        # look for subject.json and data_description.json files
        ecephys_session_folder = ecephys_session_folders[0]
        subject_json = ecephys_session_folder / "subject.json"
        subject_id = "undefined"
        if subject_json.is_file():
            subject_data = json.load(open(subject_json, "r"))
            subject_id = subject_data["subject_id"]

        data_description_json = ecephys_session_folder / "data_description.json"
        session_name = "undefined"
        if data_description_json.is_file():
            data_description = json.load(open(data_description_json, "r"))
            session_name = data_description["name"]

        log.setup_logging(
            "Job Dispatch Ecephys",
            subject_id=subject_id,
            asset_name=session_name,
        )
        aind_log_setup = True
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    logging.info(f"Running job dispatcher with the following parameters:")
    logging.info(f"\tSPLIT SEGMENTS: {SPLIT_SEGMENTS}")
    logging.info(f"\tSPLIT GROUPS: {SPLIT_GROUPS}")
    logging.info(f"\tDEBUG: {DEBUG}")
    logging.info(f"\tDEBUG DURATION: {DEBUG_DURATION}")
    logging.info(f"\tSKIP TIMESTAMPS CHECK: {SKIP_TIMESTAMPS_CHECK}")
    logging.info(f"\tMULTI SESSION: {MULTI_SESSION}")
    logging.info(f"\tINPUT: {INPUT}")
    logging.info(f"\tMIN_RECORDING_DURATION: {MIN_RECORDING_DURATION}")
    if INPUT == "spikeinterface":
        logging.info(f"\tSPIKEINTERFACE_INFO: {spikeinterface_info}")


    logging.info(f"Parsing {INPUT} input folder")
    recording_dict = {}
    include_annotations = False
    if INPUT == "aind":
        for ecephys_session_folder in ecephys_session_folders:
            session_name = None
            if (ecephys_session_folder / "data_description.json").is_file():
                data_description = json.load(open(ecephys_session_folder / "data_description.json", "r"))
                session_name = data_description["name"]

            # in the AIND pipeline, the session folder is mapped to
            ecephys_base_folder = ecephys_session_folder / "ecephys"

            if (ecephys_base_folder / "ecephys_compressed").is_dir():
                new_format = True
                ecephys_folder = ecephys_base_folder
            else:
                new_format = False
                ecephys_folder = ecephys_session_folder

            compressed = False
            if (ecephys_folder / "ecephys_compressed").is_dir():
                # most recent folder organization
                compressed = True
                ecephys_compressed_folder = ecephys_folder / "ecephys_compressed"
                ecephys_openephys_folder = ecephys_folder / "ecephys_clipped"
            else:
                # uncompressed data
                ecephys_openephys_folder = ecephys_base_folder

            logging.info(f"\tSession name: {session_name}")
            logging.info(f"\tOpen Ephys folder: {str(ecephys_openephys_folder)}")
            if compressed:
                logging.info(f"\tZarr compressed folder: {str(ecephys_compressed_folder)}")

            # get blocks/experiments and streams info
            num_blocks = se.get_neo_num_blocks("openephysbinary", ecephys_openephys_folder)
            stream_names, stream_ids = se.get_neo_streams("openephysbinary", ecephys_openephys_folder)

            # load first stream to map block_indices to experiment_names
            rec_test = se.read_openephys(ecephys_openephys_folder, block_index=0, stream_name=stream_names[0])
            record_node = list(rec_test.neo_reader.folder_structure.keys())[0]
            experiments = rec_test.neo_reader.folder_structure[record_node]["experiments"]
            exp_ids = list(experiments.keys())
            experiment_names = [experiments[exp_id]["name"] for exp_id in sorted(exp_ids)]

            logging.info(f"\tNum. Blocks {num_blocks} - Num. streams: {len(stream_names)}")
            for block_index in range(num_blocks):
                for stream_name in stream_names:
                    # skip NIDAQ and NP1-LFP streams
                    if "NI-DAQ" not in stream_name and "LFP" not in stream_name and "Rhythm" not in stream_name:
                        experiment_name = experiment_names[block_index]
                        exp_stream_name = f"{experiment_name}_{stream_name}"
                        if not compressed:
                            recording = se.read_openephys(
                                ecephys_openephys_folder, stream_name=stream_name, block_index=block_index
                            )
                        else:
                            recording = si.read_zarr(ecephys_compressed_folder / f"{exp_stream_name}.zarr")
                        recording_name = f"{exp_stream_name}_recording"

                        # fix probe information in case of missing names
                        updated_probe = None
                        probes_info = recording.get_annotation("probes_info")
                        if probes_info is not None and len(probes_info) == 1:
                            probe_info = probes_info[0]
                            probe_name = probe_info["name"]
                            if probe_name == "":
                                record_node, oe_stream_name = stream_name.split("#")
                                logging.info(
                                    f"\t\tProbe name is missing for block {block_index} - {oe_stream_name}! "
                                    "Parsing Open Ephys settings to load up-to-date probe info"
                                )
                                if block_index == 0:
                                    settings_name = "settings.xml"
                                else:
                                    settings_name = f"settings_{block_index + 1}.xml"
                                updated_probe = pi.read_openephys(
                                    ecephys_openephys_folder / record_node / settings_name,
                                    stream_name=oe_stream_name
                                )
                                recording.set_probe(updated_probe, in_place=True)
                                # make sure we the updated annotations when dumping the dict!
                                include_annotations = True

                        recording_dict[(session_name, recording_name)] = {}
                        recording_dict[(session_name, recording_name)]["input_folder"] = ecephys_session_folder
                        recording_dict[(session_name, recording_name)]["raw"] = recording

                        # load the associated LF stream (if available)
                        if "AP" in stream_name:
                            stream_name_lf = stream_name.replace("AP", "LFP")
                            exp_stream_name_lf = exp_stream_name.replace("AP", "LFP")
                            try:
                                if not compressed:
                                    recording_lf = se.read_openephys(
                                        ecephys_openephys_folder, stream_name=stream_name_lf, block_index=block_index
                                    )
                                else:
                                    recording_lf = si.read_zarr(ecephys_compressed_folder / f"{exp_stream_name_lf}.zarr")
                                if updated_probe is not None:
                                    recording_lf.set_probe(updated_probe, in_place=True)
                                recording_dict[(session_name, recording_name)]["lfp"] = recording_lf
                            except:
                                logging.info(f"\t\tNo LFP stream found for {exp_stream_name}")

    elif INPUT == "spikeglx":
        # get blocks/experiments and streams info
        spikeglx_folders = [p for p in data_folder.iterdir() if p.is_dir()]
        if len(spikeglx_folders) == 0:
            raise Exception("No valid SpikeGLX folder found.")
        elif len(spikeglx_folders) > 1 and not MULTI_SESSION:
            raise Exception("Multiple SpikeGLX sessions found in the data folder. Please only add one at a time")
    
        for spikeglx_folder in spikeglx_folders:
            session_name = spikeglx_folder.name
            stream_names, stream_ids = se.get_neo_streams("spikeglx", spikeglx_folder)
    
            # spikeglx has only one block
            num_blocks = 1
            block_index = 0
    
            logging.info(f"\tSession name: {session_name}")
            logging.info(f"\tNum. streams: {len(stream_names)}")
            logging.info(f"\tStreams detected: {stream_names}")
    
            added_any = False
            for stream_name in stream_names:
                # allow NIDQ; still skip LF and SYNC helper streams
                if ("lf" in stream_name) or ("SYNC" in stream_name):
                    logging.info(f"\tSkipping stream {stream_name} (LF/SYNC)")
                    continue
    
                try:
                    recording = se.read_spikeglx(spikeglx_folder, stream_name=stream_name)                        
    
                    # OPTIONAL: for NIDQ, keep only MN channels (drop MA) to avoid gain mismatch
                    # Your meta says snsMnMaXaDw=32,4,0,0 → the first 32 saved channels are MN
                    if stream_name == "nidq":
                        import numpy as np
                        # YOUR superficial->deep order (indices are saved MN channels 1..32)
                        order = [13, 10, 12, 14, 2, 15, 32, 4, 6, 8, 9, 11, 5, 3, 26, 24, 30, 28, 31, 29, 16, 17, 18, 1, 25, 19, 27, 21, 20, 22, 23]  # <-- 31 entries; we’ll place any missing at the end
                        
                        N = recording.get_num_channels()   # expect 32
                        saved_nums = np.arange(1, N + 1, dtype=int)  # 1..32 in saved order
                        rank = {ch: i for i, ch in enumerate(order)}
                        missing = [ch for ch in saved_nums if ch not in rank]  # e.g., 7 if it’s truly present
                        start = len(rank)
                        for k, ch in enumerate(missing):
                        rank[ch] = start + k  # append missing at the deepest end
                        
                        PITCH_UM = 20.0  # change if your probe pitch is different
                        y = np.array([rank[ch] * PITCH_UM for ch in saved_nums], dtype=float)
                        x = np.zeros(N, dtype=float)  # linear probe → x=0
                        rec_locs = np.c_[x, y]
                        recording.set_channel_locations(rec_locs)
                        logging.info(f"\tNIDQ: injected locations for {N} channels (pitch={PITCH_UM} µm). Missing: {missing}")
                        try:
                            if recording.get_num_channels() >= 36:
                                recording = recording.channel_slice(channel_ids=recording.channel_ids[:32])
                                logging.info("\tNIDQ: keeping first 32 MN channels (dropping MA)")
                        except Exception as e:
                            logging.info(f"\tNIDQ channel slice skipped (non-fatal): {e}")
    
                    recording_name = f"block{block_index}_{stream_name}_recording"
                    recording_dict[(session_name, recording_name)] = {}
                    recording_dict[(session_name, recording_name)]["input_folder"] = spikeglx_folder
                    recording_dict[(session_name, recording_name)]["raw"] = recording
                    added_any = True
    
                    # load the associated LF stream (if available) only for AP streams
                    if "ap" in stream_name:
                        stream_name_lf = stream_name.replace("ap", "lf")
                        try:
                            recording_lf = se.read_spikeglx(spikeglx_folder, stream_name=stream_name_lf)
                            recording_dict[(session_name, recording_name)]["lfp"] = recording_lf
                        except Exception:
                            logging.info(f"\t\tNo LFP stream found for {stream_name}")
    
                except Exception as e:
                    logging.info(f"\tFailed to read stream {stream_name}: {e}")
    
            # Fallback: if nothing was added (e.g., only LF detected), try to read 'nidq' explicitly
            if not added_any:
                try:
                    logging.info("\tNo eligible streams added; trying explicit 'nidq' fallback")
                    recording = se.read_spikeglx(spikeglx_folder, stream_name="nidq")
                    try:
                        if recording.get_num_channels() >= 36:
                            recording = recording.channel_slice(channel_ids=recording.channel_ids[:32])
                            logging.info("\tNIDQ fallback: keeping first 32 MN channels (dropping MA)")
                    except Exception as e:
                        logging.info(f"\tNIDQ fallback channel slice skipped (non-fatal): {e}")
    
                    recording_name = f"block{block_index}_nidq_recording"
                    recording_dict[(session_name, recording_name)] = {}
                    recording_dict[(session_name, recording_name)]["input_folder"] = spikeglx_folder
                    recording_dict[(session_name, recording_name)]["raw"] = recording
                    added_any = True
                except Exception as e:
                    logging.info(f"\tExplicit NIDQ fallback failed: {e}")


    elif INPUT == "openephys":
        # get blocks/experiments and streams info
        openephys_folders = [p for p in data_folder.iterdir() if p.is_dir()]
        if len(openephys_folders) == 0:
            raise Exception("No valid Open Ephys folder found.")
        elif len(openephys_folders) > 1:
            if not MULTI_SESSION:
                raise Exception("Multiple Open Ephys sessions found in the data folder. Please only add one at a time")

        for openephys_folder in openephys_folders:
            session_name = openephys_folder.name
            num_blocks = se.get_neo_num_blocks("openephysbinary", openephys_folder)
            stream_names, stream_ids = se.get_neo_streams("openephysbinary", openephys_folder)

            logging.info(f"\tSession name: {session_name}")
            logging.info(f"\tNum. Blocks {num_blocks} - Num. streams: {len(stream_names)}")

            # load first stream to map block_indices to experiment_names
            rec_test = se.read_openephys(openephys_folder, block_index=0, stream_name=stream_names[0])
            record_node = list(rec_test.neo_reader.folder_structure.keys())[0]
            experiments = rec_test.neo_reader.folder_structure[record_node]["experiments"]
            exp_ids = list(experiments.keys())
            experiment_names = [experiments[exp_id]["name"] for exp_id in sorted(exp_ids)]

            for block_index in range(num_blocks):
                for stream_name in stream_names:
                    if "NI-DAQ" not in stream_name and "LFP" not in stream_name and "SYNC" not in stream_name:
                        experiment_name = experiment_names[block_index]
                        exp_stream_name = f"{experiment_name}_{stream_name}"
                        recording = se.read_openephys(
                            openephys_folder, load_sync_timestamps=True, stream_name=stream_name, block_index=block_index
                        )
                        recording_name = f"{exp_stream_name}_recording"
                        recording_dict[(session_name, recording_name)] = {}
                        recording_dict[(session_name, recording_name)]["input_folder"] = openephys_folder
                        recording_dict[(session_name, recording_name)]["raw"] = recording

                        # load the associated LFP stream (if available)
                        if "AP" in stream_name:
                            stream_name_lf = stream_name.replace("AP", "LFP")
                            try:
                                recording_lf = se.read_openephys(
                                    openephys_folder, stream_name=stream_name_lf, block_index=block_index
                                )
                                recording_dict[(session_name, recording_name)]["lfp"] = recording_lf
                            except:
                                logging.info(f"\t\tNo LFP stream found for {stream_name}")

    elif INPUT == "nwb":
        # get blocks/experiments and streams info
        all_input_folders = [p for p in data_folder.iterdir() if p.is_dir()]
        if len(all_input_folders) == 1:
            nwb_files = [p for p in all_input_folders[0].iterdir() if p.name.endswith(".nwb")]
        else:
            nwb_files = [p for p in data_folder.iterdir() if p.name.endswith(".nwb")]
        logging.info(f"nwb_files: {nwb_files}")
        if len(nwb_files) == 0:
            raise ValueError("No NWB files found in the data folder")
        elif len(nwb_files) > 1:
            if not MULTI_SESSION:
                raise ValueError("Multiple NWB files found in the data folder. Please only add one at a time")

        for nwb_file in nwb_files:
            nwb_file = nwb_file.absolute()
            print(f"Processing NWB file: {nwb_file}")
            session_name = nwb_file.stem

            # spikeglx has only one block
            num_blocks = 1
            block_index = 0

            electrical_series_paths = se.NwbRecordingExtractor.fetch_available_electrical_series_paths(nwb_file)

            logging.info(f"\tSession name: {session_name}")
            logging.info(f"\tNum. Blocks {num_blocks} - Num. streams: {len(electrical_series_paths)}")
            for electrical_series_path in electrical_series_paths:
                # only use paths in acquisition
                if "acquisition" in electrical_series_path:
                    stream_name = electrical_series_path.replace("/", "-")
                    recording = se.read_nwb_recording(nwb_file, electrical_series_path=electrical_series_path)
                    if recording.sampling_frequency < 10000:
                        logging.info(
                            f"\t\t{electrical_series_path} is probably an LFP signal (sampling frequency: "
                            f"{recording.sampling_frequency} Hz). Skipping"
                        )
                        continue
                    recording_name = f"block{block_index}_{stream_name}_recording"
                    recording_dict[(session_name, recording_name)] = {}
                    recording_dict[(session_name, recording_name)]["raw"] = recording

    elif INPUT == "spikeinterface":
        from spikeinterface.extractors import recording_extractor_full_dict
        from spikeinterface.extractors.neoextractors import neo_recording_class_dict
        from probeinterface import read_probeinterface

        if isinstance(spikeinterface_info, dict):
            # spikeinterface_info already provided as a dict with a JSON file/string
            pass
        elif Path(spikeinterface_info).is_file():
            with open(spikeinterface_info, "r") as f:
                spikeinterface_info = json.load(f)
        elif isinstance(spikeinterface_info, str):
            spikeinterface_info = json.loads(spikeinterface_info)

        reader_type = spikeinterface_info.get("reader_type", None)
        assert reader_type is not None, "Reader type is required"
        assert reader_type in recording_extractor_full_dict, f"Reader type {reader_type} not supported"
        reader_kwargs = spikeinterface_info.get("reader_kwargs", None)
        keep_stream_substrings = spikeinterface_info.get("keep_stream_substrings", None)
        skip_stream_substrings = spikeinterface_info.get("skip_stream_substrings", None)
        probe_paths = spikeinterface_info.get("probe_paths", None)
        session_names = spikeinterface_info.get("session_names", None)

        if keep_stream_substrings is not None:
            assert skip_stream_substrings is None, "You cannot use both keep_stream_substrings and skip_stream_substrings"
            if isinstance(keep_stream_substrings, str):
                keep_stream_substrings = [keep_stream_substrings]

        if skip_stream_substrings is not None:
            assert keep_stream_substrings is None, "You cannot use both keep_stream_substrings and skip_stream_substrings"
            if isinstance(skip_stream_substrings, str):
                skip_stream_substrings = [skip_stream_substrings]

        # check if it's a neo reader
        if isinstance(reader_kwargs, dict):
            reader_kwargs_list = [reader_kwargs]
        elif isinstance(reader_kwargs, list):
            reader_kwargs_list = reader_kwargs
        else:
            raise ValueError("reader_kwargs should be a dict or a list of dicts")

        if len(reader_kwargs_list) > 1:
            if not MULTI_SESSION:
                raise ValueError("To use multiple sessions, you need to set the multi_session flag to True")

        if session_names is not None:
            if not isinstance(session_names, list):
                session_names = [session_names]
            if len(session_names) != len(reader_kwargs_list):
                raise ValueError(
                    "If you provide multiple session names, you need to provide one for each reader_kwargs"
                )
        else:
            session_names = [f"session{i}" for i in range(len(reader_kwargs_list))]

        if probe_paths is not None:
            if isinstance(probe_paths, (str, dict)):
                probe_paths = [probe_paths] * len(reader_kwargs_list)
            if len(probe_paths) != len(reader_kwargs_list):
                raise ValueError("If you provide multiple probe paths, you need to provide one for each reader_kwargs")
        else:
            probe_paths = [None] * len(reader_kwargs_list)

        for probe_paths_session, session_name, reader_kwargs in zip(probe_paths, session_names, reader_kwargs_list):
            if recording_extractor_full_dict[reader_type] in neo_recording_class_dict:
                num_blocks = se.get_neo_num_blocks(reader_type, **reader_kwargs)
                stream_names, stream_ids = se.get_neo_streams(reader_type, **reader_kwargs)
            else:
                num_blocks = 1
                stream_names = [None]
            for block_index in range(num_blocks):
                for stream_name in stream_names:
                    if stream_name is not None:
                        if keep_stream_substrings is not None:
                            if not any(s in stream_name for s in keep_stream_substrings):
                                logging.info(f"\tSkipping stream {stream_name} (keep substrings: {keep_stream_substrings})")
                                continue
                        if skip_stream_substrings is not None:
                            if any(s in stream_name for s in skip_stream_substrings):
                                logging.info(f"\tSkipping stream {stream_name} (skip substrings: {skip_stream_substrings})")
                                continue
                        reader_kwargs["stream_name"] = stream_name
                    logging.info(f"\tStream name: {stream_name}")
                    recording = recording_extractor_full_dict[reader_type](**reader_kwargs)
                    probe_path = None
                    if probe_paths_session is not None:
                        if isinstance(probe_paths_session, str):
                            probe_path = probe_paths_session
                        elif isinstance(probe_paths_session, dict):
                            probe_path = probe_paths_session.get(stream_name, None)
                    if probe_path is not None:
                        probegroup = read_probeinterface(probe_path)
                        recording = recording.set_probegroup(probegroup)
                    probegroup = recording.get_probegroup()
                    assert probegroup is not None, (
                        f"Probe not specified for {stream_name}. "
                        f"Use the 'probe_paths' field of spikeinterface-info to specify it."
                    )
                    recording_name = f"block{block_index}_{stream_name}_recording"
                    recording_dict[(session_name, recording_name)] = {}
                    recording_dict[(session_name, recording_name)]["raw"] = recording

    # populate job dict list
    job_dict_list = []
    logging.info("Recording to be processed in parallel:")
    for session_recording_name in recording_dict:
        session_name, recording_name = session_recording_name
        input_folder = recording_dict[session_recording_name].get("input_folder")
        recording = recording_dict[session_recording_name]["raw"]
        recording_lfp = recording_dict[session_recording_name].get("lfp", None)

        if MIN_RECORDING_DURATION != -1:
            duration = recording.get_total_duration()
            if duration < MIN_RECORDING_DURATION:
                logging.info(
                    f"\tSkipping recording {session_name}-{recording_name} with duration {np.round(duration, 2)}s"
                )
                continue

        HAS_LFP = recording_lfp is not None
        if not SPLIT_SEGMENTS:
            recordings = [recording]
            recordings_lfp = [recording_lfp] if HAS_LFP else None
        else:
            recordings = si.split_recording(recording)
            recordings_lfp = si.split_recording(recording_lfp) if HAS_LFP else None

        for recording_index, recording in enumerate(recordings):
            if SPLIT_SEGMENTS:
                recording_name_segment = f"{recording_name}{recording_index + 1}"
            else:
                recording_name_segment = f"{recording_name}"

            if HAS_LFP:
                recording_lfp = recordings_lfp[recording_index]

            # timestamps should be monotonically increasing, but we allow for small glitches
            skip_times = False
            if not SKIP_TIMESTAMPS_CHECK:
                for segment_index in range(recording.get_num_segments()):
                    times = recording.get_times(segment_index=segment_index)
                    times_diff_ms = np.diff(times) * 1000
                    num_negative_times = np.sum(times_diff_ms < -ACCEPTED_NEGATIVE_DEVIATION_MS)

                    if num_negative_times > MAX_NUM_NEGATIVE_TIMESTAMPS:
                        logging.info(
                            f"\t{recording_name}:\n\t\tSkipping timestamps for too many negative "
                            f"timestamps diffs below {ACCEPTED_NEGATIVE_DEVIATION_MS}: {num_negative_times}"
                        )
                        skip_times = True
                        break
                    max_time_diff_ms = np.max(np.abs(times_diff_ms))
                    if max_time_diff_ms > ABS_MAX_TIMESTAMPS_DEVIATION_MS:
                        logging.info(
                            f"\t{recording_name}:\n\t\tSkipping timestamps for too large time diff deviation: {max_time_diff_ms} ms"
                        )
                        skip_times = True
                        break

            if skip_times:
                recording.reset_times()

            if DEBUG:
                recording_list = []
                for segment_index in range(recording.get_num_segments()):
                    recording_one = si.split_recording(recording)[segment_index]
                    recording_one = recording_one.frame_slice(
                        start_frame=0,
                        end_frame=min(
                            int(DEBUG_DURATION * recording.sampling_frequency), recording_one.get_num_samples()
                        ),
                    )
                    recording_list.append(recording_one)
                recording = si.append_recordings(recording_list)
                if HAS_LFP:
                    recording_lfp_list = []
                    for segment_index in range(recording_lfp.get_num_segments()):
                        recording_lfp_one = si.split_recording(recording_lfp)[segment_index]
                        recording_lfp_one = recording_lfp_one.frame_slice(
                            start_frame=0,
                            end_frame=min(
                                int(DEBUG_DURATION * recording_lfp.sampling_frequency),
                                recording_lfp_one.get_num_samples(),
                            ),
                        )
                        recording_lfp_list.append(recording_lfp_one)
                    recording_lfp = si.append_recordings(recording_lfp_list)

            duration = np.round(recording.get_total_duration(), 2)

            # if multiple channel groups, process in parallel
            if SPLIT_GROUPS and len(np.unique(recording.get_channel_groups())) > 1:
                for group_name, recording_group in recording.split_by("group").items():
                    recording_name_group = f"{recording_name_segment}_group{group_name}"
                    job_dict = dict(
                        session_name=session_name,
                        recording_name=str(recording_name_group),
                        recording_dict=recording_group.to_dict(recursive=True, relative_to=data_folder),
                        skip_times=skip_times,
                        duration=duration,
                        input_folder=input_folder,
                        debug=DEBUG,
                    )
                    rec_str = f"\t{recording_name_group}\n\t\tDuration {duration} s - Num. channels: {recording_group.get_num_channels()}"
                    if HAS_LFP:
                        recording_lfp_group = recording_lfp.split_by("group")[group_name]
                        job_dict["recording_lfp_dict"] = recording_lfp_group.to_dict(
                            recursive=True, relative_to=data_folder
                        )
                        rec_str += f" (with LFP stream)"
                    logging.info(rec_str)
                    job_dict_list.append(job_dict)
            else:
                job_dict = dict(
                    session_name=session_name,
                    recording_name=str(recording_name_segment),
                    recording_dict=recording.to_dict(recursive=True, include_annotations=include_annotations, relative_to=data_folder),
                    skip_times=skip_times,
                    duration=duration,
                    input_folder=input_folder,
                    debug=DEBUG,
                )
                rec_str = f"\t{recording_name_segment}\n\t\tDuration: {duration} s - Num. channels: {recording.get_num_channels()}"
                if HAS_LFP:
                    job_dict["recording_lfp_dict"] = recording_lfp.to_dict(recursive=True, relative_to=data_folder)
                    rec_str += f" (with LFP stream)"
                logging.info(rec_str)
                job_dict_list.append(job_dict)

    if not results_folder.is_dir():
        results_folder.mkdir(parents=True)

    if MULTI_SESSION:
        logging.info("Adding session name to recording name")

    for i, job_dict in enumerate(job_dict_list):
        job_dict["multi_input"] = MULTI_SESSION
        if MULTI_SESSION:
            # we use double _ here for easy parsing
            recording_name = f"{job_dict['session_name']}__{job_dict['recording_name']}"
            job_dict["recording_name"] = recording_name
        with open(results_folder / f"job_{i}.json", "w") as f:
            json.dump(job_dict, f, indent=4, cls=SIJsonEncoder)
    logging.info(f"Generated {len(job_dict_list)} job config files")
