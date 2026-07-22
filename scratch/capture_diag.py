import sys
import mss
import cv2
import numpy as np

try:
    import dxcam
except ImportError:
    dxcam = None

print("=== Capture Diagnostics ===")
print("Python:", sys.version)

with mss.mss() as sct:
    print("\nMSS Monitor Layout:")
    for idx, mon in enumerate(sct.monitors):
        print(f"  Monitor {idx}: {mon}")

# Test coordinates from your calibration
coords = (1966, 1061, 2305, 1439)
L, T, R, B = coords
w, h = R - L, B - T

print(f"\nTarget Region: {coords} (Size: {w}x{h})")

# Test MSS capture
print("\nTesting MSS grab...")
try:
    with mss.mss() as sct:
        shot = sct.grab({"left": L, "top": T, "width": w, "height": h})
        img = np.asarray(shot)
        print(f"  MSS Success! Captured shape: {img.shape}")
        cv2.imwrite("test_mss.png", img)
        print("  Saved test_mss.png")
except Exception as e:
    print(f"  MSS Failed: {e}")

# Test DXCAM
if dxcam:
    print("\nTesting DXCAM...")
    try:
        info = dxcam.output_info()
        print(f"  DXCAM Output Info: {info}")
        
        # Test monitor 0
        try:
            cam0 = dxcam.create(output_idx=0, output_color="BGR")
            frame0 = cam0.grab(region=coords)
            print(f"  DXCAM Monitor 0 region grab: {'Success' if frame0 is not None else 'Failed (returned None)'}")
            if frame0 is not None:
                print(f"    Shape: {frame0.shape}")
                cv2.imwrite("test_dxcam_0.png", frame0)
            cam0.release()
        except Exception as e:
            print(f"  DXCAM Monitor 0 error: {e}")
            
        # Test monitor 1
        try:
            cam1 = dxcam.create(output_idx=1, output_color="BGR")
            frame1 = cam1.grab(region=coords)
            print(f"  DXCAM Monitor 1 region grab: {'Success' if frame1 is not None else 'Failed (returned None)'}")
            if frame1 is not None:
                print(f"    Shape: {frame1.shape}")
                cv2.imwrite("test_dxcam_1.png", frame1)
            cam1.release()
        except Exception as e:
            print(f"  DXCAM Monitor 1 error: {e}")
            
    except Exception as e:
        print(f"  DXCAM General Setup Failed: {e}")
else:
    print("\nDXCAM not installed or available.")
