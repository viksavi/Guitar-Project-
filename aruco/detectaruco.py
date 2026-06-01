import cv2
import numpy as np
import os
import trimesh
import json

# define paths
frames_path = "../dataset/deux/frames/2/video_a_p_s" # resize 
obj_path = "../FoundationPose/demo_data/guitar/mesh/textured_simple.obj"
obj_path = "./Untitled1.obj"
obj_path = "./guitar.obj"

"""
mtx = np.array([[1264.0, 0.0, 1404.8],
       [0.0, 1264.7, 749.7],
       [0.0, 0.0, 1.0]], dtype=np.float32)
dist = np.zeros((5, 1)) # lens distortion 
"""
mtx = np.array([[1283.7, 0.0, 1363.2],
       [0.0, 1289.8, 761.7],
       [0.0, 0.0, 1.0]], dtype=np.float32)
       
# dist = np.array((0.36739806, -0.01736701, 0.6391675, -2.59473474, 0.0)) # lens distortion 
dist = np.zeros((5,1))


marker_size = 0.03  # 3 centimeters

marker_0_3d = np.array([
    [-marker_size / 2, marker_size / 2, 0],
    [ marker_size / 2, marker_size / 2, 0],
    [ marker_size / 2, -marker_size / 2, 0],
    [-marker_size / 2, -marker_size / 2, 0]
], dtype=np.float32)

# Marker 10 is shifted 20cm (0.2m) to the right on the X-axis
"""
marker_10_3d = marker_0_3d + np.array([-0.135, 0, 0], dtype=np.float32)
marker_3_3d = marker_0_3d + np.array([0.03, -0.16, 0], dtype=np.float32)
marker_1_3d = marker_3_3d + np.array([-0.16, 0.01, 0], dtype=np.float32)
marker_4_3d = marker_3_3d + np.array([0.002, -0.09, 0], dtype=np.float32)
marker_7_3d = marker_4_3d + np.array([0.07, -0.057, 0], dtype=np.float32)
marker_6_3d = marker_4_3d + np.array([-0.03, -0.057, 0], dtype=np.float32)
marker_5_3d = marker_6_3d + np.array([-0.023, -0.08, 0], dtype=np.float32)
"""




marker_10_3d = marker_0_3d + np.array([-0.11, 0, 0], dtype=np.float32)
marker_3_3d = marker_0_3d + np.array([0.03, -0.16, 0], dtype=np.float32)
marker_1_3d = marker_0_3d + np.array([-0.16, -0.16, 0], dtype=np.float32)
marker_5_3d = marker_0_3d + np.array([-0.025, -0.385, 0], dtype=np.float32)
marker_4_3d = marker_3_3d + np.array([0, -0.10, 0], dtype=np.float32)
marker_7_3d = marker_4_3d + np.array([0.06, -0.06, 0], dtype=np.float32)
marker_6_3d = marker_7_3d + np.array([-0.03, -0.07, 0], dtype=np.float32)


# Store them in a dictionary mapping ID -> 3D corners
marker_db = {
    0: marker_0_3d,
    10: marker_10_3d,
    3: marker_3_3d,
    1: marker_1_3d,
    5: marker_5_3d,
    4: marker_4_3d,
    7: marker_7_3d,
    6: marker_6_3d,
    }




all_frames = sorted(os.listdir(frames_path))

mesh = trimesh.load(obj_path)
# print(mesh.bounding_box.extents)
model_vertices = mesh.vertices / 1 # [0.0, -0.06, -0.05]
translation_offset = np.array([0.0, -0.0, -0.0]) # Adjust in meters
model_vertices = model_vertices + translation_offset


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)


out_data = {}
frame_n = 1

render = False

for frame in all_frames:
    image = cv2.imread(frames_path + "/" + frame)
    corners, marker_ids, rejected_candidates = detector.detectMarkers(image)

    # Arrays to aggregate data from all detected visible markers
    all_obj_pts = []
    all_img_pts = []

    if corners:

        for corner, marker_id in zip(corners, marker_ids):
            m_id = marker_id[0]
            
            # Check if this is a marker we have measured/defined in our DB
            if m_id in marker_db :
                # Add the 3D real-world coordinates of this marker's corners
                all_obj_pts.append(marker_db[m_id])
                # Add the 2D image coordinates found by OpenCV
                all_img_pts.append(corner.reshape(4, 2))
                
                # Optional: Draw the marker outline for debugging
                cv2.polylines(image, [corner.astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)

    # If we found at least one of our known markers, estimate the unified pose
    if len(all_obj_pts) > 0:
        # Concatenate lists into continuous numpy arrays
        np_obj_pts = np.vstack(all_obj_pts).astype(np.float32)
        np_img_pts = np.vstack(all_img_pts).astype(np.float32)

        # Solve PnP using all visible markers simultaneously
        success, rvec, tvec = cv2.solvePnP(np_obj_pts, np_img_pts, mtx, dist)

        if success:
            out_data[frame_n] = ([rvec[0][0], rvec[1][0], rvec[2][0]],
                                [tvec[0][0], tvec[1][0], tvec[2][0]])
            frame_n += 1

            if render:
                # Project the .obj vertices onto the 2D screen
                # Note: If your guitar model's origin doesn't align with Marker 0's origin,
                # you may need to add a translation/rotation offset to model_vertices here.
                img_pts_mesh, _ = cv2.projectPoints(model_vertices, rvec, tvec, mtx, dist)
                img_pts_mesh = np.int32(img_pts_mesh).reshape(-1, 2)
                                
                # Render the Mesh Wireframe (First 500 faces)
                mesh.fix_normals()

                for i, face in enumerate(mesh.faces):
                    # Get the 3D normal of the face
                    normal = mesh.face_normals[i]
                    
                    # Get one vertex of the face to calculate the view vector
                    vertex = model_vertices[face[0]]
                    
                    # In camera space, the vector from camera (0,0,0) to vertex:
                    # Transform vertex to camera coordinate system first
                    R, _ = cv2.Rodrigues(rvec)
                    vertex_cam = R.dot(vertex) + tvec.squeeze()
                    
                    # Transform normal to camera coordinate system
                    normal_cam = R.dot(normal)
                    
                    # Ray from camera to vertex is just vertex_cam
                    # If dot product is >= 0, the face points away from the camera (Backface culling)
                    if np.dot(normal_cam, vertex_cam) < 0:
                        pts = img_pts_mesh[face]
                        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)  
                # Draw camera coordinate axes at the defined global origin (Marker 0)
                cv2.drawFrameAxes(image, mtx, dist, rvec, tvec, 0.1)

                cv2.imwrite("./arucoframes/" + frame, image)
    if frame_n % 100 == 0:
        print("Frame", frame)
        with open("outdata.json", "w") as f:
            f.write(json.dumps(out_data))

cv2.waitKey(0)
cv2.destroyAllWindows()

