import pyrealsense2 as rs

ctx = rs.context()
for dev in ctx.devices:
    print('Camera:', dev.get_info(rs.camera_info.name))
    print('USB:   ', dev.get_info(rs.camera_info.usb_type_descriptor))
    for s in dev.query_sensors():
        for p in s.get_stream_profiles():
            v = p.as_video_stream_profile()
            if v.fps() == 30 and v.width() in (640, 848, 1280):
                print(
                    f'  {p.stream_name():8} {v.width()}x{v.height()} {p.format()} @ {v.fps()}fps'
                )
