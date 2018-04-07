# Training Data

We provide a small dataset collected by Intel Realsense for quick testing:
- realsense.zip: http://deepcompletion.cs.princeton.edu/public/data/realsense.zip      

Here is a list of data you may get upon SUNCG/Matterport3D/ScanNet access:
- matterport_render_depth.zip: The depth rendered from mesh reconstruction for scenes from Matterport3D. The images are 16bit PNG with 4000 x depth in meters. (Require access to Matterport3D)
- matterport_render_normal.zip: The surface normal for the rendered depth from Matterport3D. The image are 16bit PNG with [0,65535] mapping to [-1,+1] for three directions. (Require access to Matterport3D)
- scannet_render_depth.zip: The depth rendered from mesh reconstruction for scenes from ScanNet. The same as matterport. (Require access to ScanNet)
- scannet_render_normal.zip: The surface normal for the rendered depth from ScanNet. The same as matterport. (Require access to ScanNet)
- pbrs_boundary.zip: The boundary ground truth for physically based rendering in SUNCG-RGBD dataset. The images are 16bit PNG with 0 for no boundary, 1 for occlusion, 2 for crease. (Require access to SUNCG)