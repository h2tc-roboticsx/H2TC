
import cv2
import numpy as np
import os.path as osp

image_points = []

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        # print(xy)  
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=3)  # darw on the image
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=1)
        image_points.append(x)
        image_points.append(y)
        cv2.imshow("image", img)

def cam_extri_estimation(save_folder):
    cam_K = np.array([[693.91839599609375, 0.0, 665.73150634765625, 0.0],
                            [0.0, 693.91839599609375, 376.775787353515625, 0.0],
                            [0.0, 0.0, 1.0, 0.0]])
    model_points = np.array([[0.0,  0.0,  0.0],
                             [2.0,  0.0,  0.0],
                             [2.0,  0.0,  5.0],
                             [0.0,  0.0,  5.0]])
    
    image_points_np = np.array(image_points).astype(np.float32)
    image_points_np = image_points_np.reshape((4,2))
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = \
            cv2.solvePnP(model_points, image_points_np, cam_K[:3,:3], dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
    rotM = cv2.Rodrigues(rotation_vector)[0]
    cam_RT = np.concatenate((rotM,translation_vector),axis=1)
    t_vec = np.array([[0,0,0,1]])
    RT_homo = np.concatenate((cam_RT,t_vec),axis=0)
    
    # save camera files
    np.savetxt(save_folder + "/CamExtr.txt", RT_homo)
    np.savetxt(save_folder + "/CamIntr.txt", cam_K)
    print("save to %s", (save_folder + "/CamExtr.txt"))

if __name__ == '__main__':
    img_file = "L:/h2tc_dataset/002870/processed/rgbd0/left_0000.jpg"
    img_file.replace("\\","/")
    
    # 1. load image 
    img = cv2.imread(img_file)
    # img = cv2.resize(img, (1280, 720))  
    cv2.namedWindow("image", 0)  
    cv2.resizeWindow("image", (1280, 720))  

    # 2. click to mark the ground corners
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)  
    cv2.imshow("image", img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

    # 3. compute the camera extrinsic parameter
    save_folder = osp.dirname(osp.dirname(osp.dirname(img_file))) 
    cam_extri_estimation(save_folder) 
