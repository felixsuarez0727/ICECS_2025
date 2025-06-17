import h5py
import numpy as np
import collections
import argparse
import sys
import time
import os

def analyze_hdf5_dataset(file_path, output_dir=None, analyze_signals=False):
    """
    Analyze HDF5 dataset and output detailed information
    
    Args:
        file_path (str): Path to the HDF5 file
        output_dir (str): Directory to save results (if None, print to console)
        analyze_signals (bool): Whether to analyze signal characteristics
    """
    try:
        print(f"Analyzing HDF5 file: {file_path}")
        start_time = time.time()
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            log_file = open(os.path.join(output_dir, 'hdf5_analysis.txt'), 'w')
            def log(msg):
                print(msg)
                log_file.write(msg + "\n")
        else:
            def log(msg):
                print(msg)
        
        with h5py.File(file_path, 'r') as hf:
            # Get all keys
            keys = list(hf.keys())
            total_signals = len(keys)
            
            log("\n" + "=" * 50)
            log(f"Total Number of Signals: {total_signals:,}")
            log("=" * 50)
            
            # Analyze keys distribution
            log("\nAnalyzing signal distribution...")
            
            # Extract all signal types (modulation and domain)
            signal_types = []
            
            # Process first 10 keys to understand structure
            log("\nSample Key Analysis:")
            for i, key in enumerate(keys[:10]):
                log(f"Key {i+1}: {key}")
                try:
                    if isinstance(key, tuple) or (isinstance(key, str) and ('(' in key)):
                        # Convert string tuple to actual tuple if necessary
                        if isinstance(key, str):
                            import ast
                            try:
                                key_tuple = ast.literal_eval(key)
                                log(f"  Parsed: {key_tuple}")
                                
                                if len(key_tuple) >= 2:
                                    mod_type = key_tuple[0]
                                    domain = key_tuple[1]
                                    log(f"  Modulation: {mod_type}, Domain: {domain}")
                                    signal_types.append((mod_type, domain))
                            except:
                                log(f"  Warning: Could not parse key: {key}")
                        else:
                            key_tuple = key
                            mod_type = key_tuple[0]
                            domain = key_tuple[1]
                            log(f"  Modulation: {mod_type}, Domain: {domain}")
                            signal_types.append((mod_type, domain))
                    else:
                        log(f"  No tuple structure detected")
                        
                    # Analyze sample signal
                    sample_signal = hf[key][:]
                    log(f"  Signal shape: {sample_signal.shape}")
                    log(f"  Signal type: {sample_signal.dtype}")
                    log(f"  Signal range: [{np.min(sample_signal):.4f}, {np.max(sample_signal):.4f}]")
                    log(f"  Signal mean: {np.mean(sample_signal):.4f}")
                    log(f"  Signal std: {np.std(sample_signal):.4f}")
                    log("  " + "-" * 30)
                except Exception as e:
                    log(f"  Error analyzing key: {str(e)}")
            
            # Full distribution analysis (by modulation and domain)
            log("\nProcessing all keys to calculate distribution...")
            all_signal_types = []
            
            for i, key in enumerate(keys):
                if i % 10000 == 0 and i > 0:
                    log(f"  Processed {i:,}/{total_signals:,} keys...")
                try:
                    if isinstance(key, tuple) or (isinstance(key, str) and ('(' in key)):
                        # Convert string tuple to actual tuple if necessary
                        if isinstance(key, str):
                            import ast
                            try:
                                key_tuple = ast.literal_eval(key)
                                if len(key_tuple) >= 2:
                                    mod_type = key_tuple[0]
                                    domain = key_tuple[1]
                                    all_signal_types.append((mod_type, domain))
                            except:
                                pass
                        else:
                            key_tuple = key
                            mod_type = key_tuple[0]
                            domain = key_tuple[1]
                            all_signal_types.append((mod_type, domain))
                except:
                    pass
            
            # Count distribution
            distribution = collections.Counter(all_signal_types)
            
            # Print distribution table
            log("\n" + "=" * 80)
            log("SIGNAL DISTRIBUTION")
            log("=" * 80)
            log(f"{'Modulation Type':<25} | {'Domain':<30} | {'Count':<10} | {'Percentage':<10}")
            log("-" * 80)
            
            for (mod_type, domain), count in sorted(distribution.items()):
                percentage = (count / total_signals) * 100
                log(f"{mod_type:<25} | {domain:<30} | {count:<10,} | {percentage:.2f}%")
            
            # Summary by modulation type
            mod_distribution = collections.Counter([mod for mod, _ in all_signal_types])
            log("\n" + "=" * 50)
            log("SUMMARY BY MODULATION TYPE")
            log("=" * 50)
            log(f"{'Modulation Type':<25} | {'Count':<10} | {'Percentage':<10}")
            log("-" * 50)
            
            for mod_type, count in sorted(mod_distribution.items()):
                percentage = (count / total_signals) * 100
                log(f"{mod_type:<25} | {count:<10,} | {percentage:.2f}%")
            
            # Summary by domain
            domain_distribution = collections.Counter([domain for _, domain in all_signal_types])
            log("\n" + "=" * 50)
            log("SUMMARY BY DOMAIN")
            log("=" * 50)
            log(f"{'Domain':<30} | {'Count':<10} | {'Percentage':<10}")
            log("-" * 50)
            
            for domain, count in sorted(domain_distribution.items()):
                percentage = (count / total_signals) * 100
                log(f"{domain:<30} | {count:<10,} | {percentage:.2f}%")
            
            # Identify AM-related signals
            am_types = ['AM-DSB', 'AM-SSB', 'ASK']
            am_signals = []
            
            for (mod_type, domain), count in distribution.items():
                if any(am_type in mod_type for am_type in am_types):
                    am_signals.append(((mod_type, domain), count))
                    
            # Display AM signals summary
            log("\n" + "=" * 50)
            log("AM SIGNALS SUMMARY")
            log("=" * 50)
            log(f"{'Modulation Type':<25} | {'Domain':<30} | {'Count':<10}")
            log("-" * 50)
            
            total_am = 0
            for (mod_type, domain), count in sorted(am_signals, key=lambda x: x[1], reverse=True):
                log(f"{mod_type:<25} | {domain:<30} | {count:<10,}")
                total_am += count
                
            log("-" * 50)
            log(f"{'Total AM Signals':<56} | {total_am:<10,}")
            
            # Signal analysis (optional)
            if analyze_signals:
                log("\n" + "=" * 50)
                log("SIGNAL CHARACTERISTICS ANALYSIS")
                log("=" * 50)
                
                # Analyze a sample of signals
                log("\nAnalyzing signal characteristics (sampling 1000 signals)...")
                
                signal_lengths = []
                signal_channels = []
                signal_means = []
                signal_stds = []
                
                # Sample signals for analysis
                np.random.seed(42)  # For reproducibility
                sample_keys = np.random.choice(keys, min(1000, len(keys)), replace=False)
                
                for i, key in enumerate(sample_keys):
                    if i % 100 == 0 and i > 0:
                        log(f"  Analyzed {i}/1000 signals...")
                    
                    try:
                        signal = hf[key][:]
                        signal_lengths.append(signal.shape[0])
                        
                        if len(signal.shape) > 1:
                            signal_channels.append(signal.shape[1])
                        else:
                            signal_channels.append(1)
                        
                        signal_means.append(np.mean(signal))
                        signal_stds.append(np.std(signal))
                    except Exception as e:
                        log(f"  Error analyzing signal {key}: {str(e)}")
                
                log("\nSignal Statistics:")
                log(f"  Average Length: {np.mean(signal_lengths):.2f} samples")
                log(f"  Length Range: {min(signal_lengths)} - {max(signal_lengths)} samples")
                
                if len(set(signal_channels)) == 1:
                    log(f"  Channels: {signal_channels[0]}")
                else:
                    log(f"  Channels: Varies ({min(signal_channels)} - {max(signal_channels)})")
                
                log(f"  Average Mean Value: {np.mean(signal_means):.6f}")
                log(f"  Average Standard Deviation: {np.mean(signal_stds):.6f}")
            
            # Dataset size estimate
            mem_estimate = 0
            sample_count = min(100, len(keys))
            
            log("\n" + "=" * 50)
            log("MEMORY USAGE ESTIMATE")
            log("=" * 50)
            
            log(f"Sampling {sample_count} signals to estimate size...")
            
            for i, key in enumerate(np.random.choice(keys, sample_count, replace=False)):
                try:
                    signal = hf[key][()]
                    mem_estimate += signal.nbytes
                except Exception as e:
                    log(f"  Error estimating size for {key}: {str(e)}")
            
            avg_signal_size = mem_estimate / sample_count
            total_estimate = avg_signal_size * total_signals
            
            log(f"Average Signal Size: {avg_signal_size / 1024:.2f} KB")
            log(f"Estimated Total Dataset Size: {total_estimate / (1024**3):.2f} GB")
            
            end_time = time.time()
            log("\n" + "=" * 50)
            log(f"Analysis completed in {end_time - start_time:.2f} seconds")
            log("=" * 50)
        
        if output_dir:
            log_file.close()
            print(f"Analysis results saved to {os.path.join(output_dir, 'hdf5_analysis.txt')}")
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze HDF5 Dataset')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to the HDF5 file')
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to save analysis results')
    parser.add_argument('--analyze_signals', action='store_true',
                        help='Analyze signal characteristics (slower)')
    
    args = parser.parse_args()
    analyze_hdf5_dataset(args.file, args.output, args.analyze_signals)