# std
import argparse
from argparse import RawTextHelpFormatter
import glob
from os import makedirs
from os.path import join, exists, basename, splitext
# 3p
import cv2
from tqdm import tqdm
# project
from exposure_enhancement import enhance_image_exposure, enhance_with_dlee


def main(args):
    
    # load images
    imdir = args.folder
    ext = ['png', 'jpg', 'bmp']    # Add image formats here
    files = []
    [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
    #[files.extend(glob.glob(join(imdir, f"*.{e}"))) for e in ext]
    images = [cv2.imread(file) for file in files]

    # if not files:
    #     print("No images found in the specified folder.")
    #     return
    
    # print(f"Files found: {files}")
    # images = [cv2.imread(file) for file in files]

    # create save directory
    directory = join(imdir, "enhanced")
    if not exists(directory):
        makedirs(directory)

    # enhance images
    for i, image in tqdm(enumerate(images), desc="Enhancing images"):
        # if image is None:
        #     print(f"Warning: Unable to read {files[i]}. Skipping.")
        #     continue
        
        enhanced_image_bgr = enhance_image_exposure(image, args.gamma, args.lambda_, not args.lime,
                                                sigma=args.sigma, bc=args.bc, bs=args.bs, be=args.be, eps=args.eps)
        enhanced_image_rgb = cv2.cvtColor(enhanced_image_bgr, cv2.COLOR_BGR2RGB)
        final_output = enhance_with_dlee(enhanced_image_rgb)

        filename = basename(files[i])
        name, ext = splitext(filename)
        method = "LIME" if args.lime else "DUAL"
        corrected_name = f"{name}_DLEE_g{args.gamma}_l{args.lambda_}{ext}"
        cv2.imwrite(join(directory, corrected_name), final_output)

    # image_bgr = cv2.imread("low_light_image.jpg")  # OpenCV loads images in BGR

    # # Step 2: Enhance exposure using DUAL/LIME
    # enhanced_image_bgr = enhance_image_exposure(image_bgr, gamma=2.2, lambda_=0.15, use_dlee=False)

    # # Step 3: Convert BGR to RGB before passing to DLEE
    # enhanced_image_rgb = cv2.cvtColor(enhanced_image_bgr, cv2.COLOR_BGR2RGB)

    # # Step 4: Pass the enhanced image to DLEE for further improvement
    # final_output = enhance_with_dlee(enhanced_image_rgb)

    # # Step 5: Save and display the final enhanced image
    # cv2.imwrite("final_enhanced_image.jpg", final_output)
    # cv2.imshow("Final Enhanced Image", final_output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python implementation of two low-light image enhancement techniques via illumination map estimation.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-f", '--folder', default='./demo/', type=str,
                        help="folder path to test images.")
    parser.add_argument("-g", '--gamma', default=0.6, type=float,
                        help="the gamma correction parameter.")
    parser.add_argument("-l", '--lambda_', default=0.15, type=float,
                        help="the weight for balancing the two terms in the illumination refinement optimization objective.")
    parser.add_argument("-ul", "--lime", action='store_true',
                        help="Use the LIME method. By default, the DUAL method is used.")
    parser.add_argument("-s", '--sigma', default=3, type=int,
                        help="Spatial standard deviation for spatial affinity based Gaussian weights.")
    parser.add_argument("-bc", default=1, type=float,
                        help="parameter for controlling the influence of Mertens's contrast measure.")
    parser.add_argument("-bs", default=1, type=float,
                        help="parameter for controlling the influence of Mertens's saturation measure.")
    parser.add_argument("-be", default=1, type=float,
                        help="parameter for controlling the influence of Mertens's well exposedness measure.")
    parser.add_argument("-eps", default=1e-3, type=float,
                        help="constant to avoid computation instability.")

    args = parser.parse_args()
    main(args)
