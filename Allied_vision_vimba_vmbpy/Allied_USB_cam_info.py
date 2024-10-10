import vmbpy

vmb = vmbpy.VmbSystem.get_instance()

camera_count = 0
with vmb:
    cams = vmb.get_all_cameras()
    for cam in cams:
        # Count camera
        camera_count += 1
        print(f"Camera count  {camera_count}:")
        
        try:
            print(f"Camera Name: {cam.get_name()}")
            print(f"Model: {cam.get_model()}")
            print(f"Serial Number: {cam.get_serial()}")
            print(f"Camera ID: {cam.get_id()}")
            print(f"Interface ID: {cam.get_interface_id()}")
        except Exception as e:
            print(f"Error accessing camera details: {e}")

if camera_count == 0:
    print("No cameras found.")
else:
    print(f"Found {camera_count} cameras.")