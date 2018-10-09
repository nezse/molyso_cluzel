import numpy as np
from matplotlib import pylab
from molyso.test import test_image
from molyso.mm.image import Image
from molyso.generic import registration, rotation

from tifffile import TiffFile
import os
from tifffile import imsave
from ..debugging import DebugPlot

def shift_molyso(image, image_2):
    shift, = registration.translation_2x1d(image, image_2)
    return shift

def create_stack(folder_image, start_inx, end_inx):
    stack = []

    for idx in range(start_inx, end_inx):
        name_image = folder_image + slash + str(idx) + ".tif"

        t = TiffFile(name_image)
        image = t.asarray()
        t.close()

        stack.append(image)

    stack = np.array(stack)
    return stack

def shift_scikit(image, image_2):
    from skimage.feature import register_translation

    shift, error, diffphase = register_translation(image, image_2)
    return [int(s) for s in shift]

def shift_scikit_subpixel(image, image_2, subpixel):
    from skimage.feature import register_translation

    shift, error, diffphase = register_translation(image, image_2, subpixel)
    return [int(s) for s in shift]

def align_subpixel_one_fov_PC_only(folder_image, start_inx, end_inx):
    new_stack_m = []

    #reference image - the first one
    name_image = folder_image + slash + "PC" +slash + str(start_inx) + ".tif"
    t = TiffFile(name_image)
    image = t.asarray()
    t.close()
    new_stack_m.append(image)

    #align al the subsequent pictures to that one using cross correlation
    for idx in range(start_inx + 1, end_inx + 1):
        name_image = folder_image + slash + "PC" +slash + str(idx) + ".tif"

        t = TiffFile(name_image)
        image_2 = t.asarray()
        t.close()


        #shift_m = shift_molyso(image, image_2)

        shift_m = shift_scikit_subpixel(image, image_2, 100)

        image_3 = registration.shift_image(image_2, shift_m, background='blank')
        new_stack_m.append(image_3)


    new_stack_m = np.array(new_stack_m)
    return new_stack_m

def align_one_fov_PC_only(folder_image, start_inx, end_inx):
    new_stack_m = []

    #reference image - the first one
    name_image = folder_image + slash + "PC" +slash + str(start_inx) + ".tif"
    t = TiffFile(name_image)
    image = t.asarray()
    t.close()
    new_stack_m.append(image)

    #align al the subsequent pictures to that one using cross correlation
    for idx in range(start_inx + 1, end_inx + 1):
        name_image = folder_image + slash + "PC" +slash + str(idx) + ".tif"

        t = TiffFile(name_image)
        image_2 = t.asarray()
        t.close()


        #shift_m = shift_molyso(image, image_2)

        shift_m = shift_scikit(image, image_2)

        image_3 = registration.shift_image(image_2, shift_m, background='blank')
        new_stack_m.append(image_3)


    new_stack_m = np.array(new_stack_m)
    return new_stack_m

def align_one_fov_PC_YFP(folder_image, start_inx, end_inx, FP ):
    new_stack_pc = []
    new_stack_yfp = []

    #reference image - the first one
    name_image_pc = folder_image + slash + "PC" +slash + str(start_inx) + ".tif"
    name_image_yfp = folder_image + slash + FP + slash + str(start_inx) + ".tif"

    t = TiffFile(name_image_pc)
    image_reference_pc = t.asarray()
    t.close()
    new_stack_pc.append(image_reference_pc)

    t = TiffFile(name_image_yfp)
    image_yfp = t.asarray()
    t.close()
    new_stack_yfp.append(image_yfp)

    #align al the subsequent pictures to that one using cross correlation
    for idx in range(start_inx + 1, end_inx + 1):

        name_image_pc = folder_image + slash + "PC" + slash + str(idx) + ".tif"
        name_image_yfp = folder_image + slash + FP + slash + str(idx) + ".tif"

        t = TiffFile(name_image_pc)
        image_pc = t.asarray()
        t.close()

        t = TiffFile(name_image_yfp)
        image_yfp = t.asarray()
        t.close()
        # new_stack_yfp.append(image_yfp)

        shift_pc = shift_scikit(image_reference_pc, image_pc)
        image_pc_shifted = registration.shift_image(image_pc, shift_pc, background='blank')
        new_stack_pc.append(image_pc_shifted)

        #shift_yfp = shift_scikit(image_reference_pc, image_yfp)
        image_yfp_shifted = registration.shift_image(image_yfp, shift_pc, background='input')
        new_stack_yfp.append(image_yfp_shifted)

    new_stack_pc = np.array(new_stack_pc)
    new_stack_yfp = np.array(new_stack_yfp)

    return [new_stack_pc, new_stack_yfp]


def align_one_fov_PC_RFP(folder_image, start_inx, end_inx ):
    new_stack_pc = []
    new_stack_rfp = []

    #reference image - the first one
    name_image_pc = folder_image + slash + "PC" +slash + str(start_inx) + ".tif"
    name_image_rfp = folder_image + slash + "RFP" + slash + str(start_inx) + ".tif"

    t = TiffFile(name_image_pc)
    image_reference_pc = t.asarray()
    t.close()
    new_stack_pc.append(image_reference_pc)

    t = TiffFile(name_image_rfp)
    image_rfp = t.asarray()
    t.close()
    new_stack_rfp.append(image_rfp)

    #align al the subsequent pictures to that one using cross correlation
    for idx in range(start_inx + 1, end_inx + 1):

        name_image_pc = folder_image + slash + "PC" + slash + str(idx) + ".tif"
        name_image_rfp = folder_image + slash + "RFP" + slash + str(idx) + ".tif"

        t = TiffFile(name_image_pc)
        image_pc = t.asarray()
        t.close()

        t = TiffFile(name_image_rfp)
        image_rfp = t.asarray()
        t.close()
        # new_stack_rfp.append(image_rfp)

        shift_pc = shift_scikit(image_reference_pc, image_pc)
        image_pc_shifted = registration.shift_image(image_pc, shift_pc, background='blank')
        new_stack_pc.append(image_pc_shifted)

        #shift_rfp = shift_scikit(image_reference_pc, image_rfp)
        image_rfp_shifted = registration.shift_image(image_rfp, shift_pc, background='input')
        new_stack_rfp.append(image_rfp_shifted)

    new_stack_pc = np.array(new_stack_pc)
    new_stack_yfp = np.array(new_stack_rfp)

    return [new_stack_pc, new_stack_rfp]



def align_and_rotate_one_fov_PC_FP(folder_image, start_inx, end_inx, FP ):
    new_stack_pc = []
    new_stack_yfp = []

    #reference image - the first one
    name_image_pc = folder_image + slash + "PC" +slash + str(start_inx) + ".tif"
    name_image_yfp = folder_image + slash + FP + slash + str(start_inx) + ".tif"

    t = TiffFile(name_image_pc)
    image_reference_pc = t.asarray()
    t.close()


    t = TiffFile(name_image_yfp)
    image_yfp = t.asarray()
    t.close()

    #Rotation reference image
    steps = 10
    smoothing_signal_length = 15
    angle = rotation.find_rotation(image_reference_pc, steps, smoothing_signal_length)

    new_image_reference_pc, angle, h, w = rotation.apply_rotate_and_cleanup(image_reference_pc, angle)
    image_yfp, angle, h, w = rotation.apply_rotate_and_cleanup(image_yfp, angle)

    new_stack_pc.append(new_image_reference_pc)
    new_stack_yfp.append(image_yfp)




    #align al the subsequent pictures to that one using cross correlation
    for idx in range(start_inx + 1, end_inx + 1):

        name_image_pc = folder_image + slash + "PC" + slash + str(idx) + ".tif"
        name_image_yfp = folder_image + slash + FP + slash + str(idx) + ".tif"

        t = TiffFile(name_image_pc)
        image_pc = t.asarray()
        t.close()

        t = TiffFile(name_image_yfp)
        image_yfp = t.asarray()
        t.close()


        #shift for aligment
        shift_pc = shift_scikit(image_reference_pc, image_pc)
        image_pc_shifted = registration.shift_image(image_pc, shift_pc, background='blank')
        image_pc_rotated = rotation.apply_rotate_and_cleanup(image_pc_shifted, angle)[0]

        new_stack_pc.append(image_pc_rotated)

        #shift_yfp = shift_scikit(image_reference_pc, image_yfp)
        image_yfp_shifted = registration.shift_image(image_yfp, shift_pc, background='input')
        image_yfp_rotated = rotation.apply_rotate_and_cleanup(image_yfp_shifted, angle)[0]

        new_stack_yfp.append(image_yfp_rotated)

    new_stack_pc = np.array(new_stack_pc)
    new_stack_yfp = np.array(new_stack_yfp)

    return [new_stack_pc, new_stack_yfp]



def align_and_rotate_one_fov_PC_RFP(folder_image, start_inx, end_inx ):
    new_stack_pc = []
    new_stack_rfp = []

    #reference image - the first one
    name_image_pc = folder_image + slash + "PC" +slash + str(start_inx) + ".tif"
    name_image_rfp = folder_image + slash + "RFP" + slash + str(start_inx) + ".tif"

    t = TiffFile(name_image_pc)
    image_reference_pc = t.asarray()
    t.close()


    t = TiffFile(name_image_rfp)
    image_rfp = t.asarray()
    t.close()

    #Rotation reference image
    steps = 10
    smoothing_signal_length = 15
    angle = rotation.find_rotation(image_reference_pc, steps, smoothing_signal_length)

    new_image_reference_pc, angle, h, w = rotation.apply_rotate_and_cleanup(image_reference_pc, angle)
    image_rfp, angle, h, w = rotation.apply_rotate_and_cleanup(image_rfp, angle)

    new_stack_pc.append(new_image_reference_pc)
    new_stack_rfp.append(image_rfp)




    #align al the subsequent pictures to that one using cross correlation
    for idx in range(start_inx + 1, end_inx + 1):

        name_image_pc = folder_image + slash + "PC" + slash + str(idx) + ".tif"
        name_image_rfp = folder_image + slash + "RFP" + slash + str(idx) + ".tif"

        t = TiffFile(name_image_pc)
        image_pc = t.asarray()
        t.close()

        t = TiffFile(name_image_rfp)
        image_rfp = t.asarray()
        t.close()


        #shift for aligment
        shift_pc = shift_scikit(image_reference_pc, image_pc)
        image_pc_shifted = registration.shift_image(image_pc, shift_pc, background='blank')
        image_pc_rotated = rotation.apply_rotate_and_cleanup(image_pc_shifted, angle)[0]

        new_stack_pc.append(image_pc_rotated)

        #shift_rfp = shift_scikit(image_reference_pc, image_rfp)
        image_rfp_shifted = registration.shift_image(image_rfp, shift_pc, background='input')
        image_rfp_rotated = rotation.apply_rotate_and_cleanup(image_rfp_shifted, angle)[0]

        new_stack_rfp.append(image_rfp_rotated)

    new_stack_pc = np.array(new_stack_pc)
    new_stack_rfp = np.array(new_stack_rfp)

    return [new_stack_pc, new_stack_rfp]


def align_and_cut_experiment(path_file, list_fov, start_index_pic, end_index_pic):

    for fov in list_fov:

        file_name = path_file + slash + str(fov)

        aligned_stack = align_one_fov_PC_only(file_name, start_index_pic, end_index_pic)
        imsave(file_name+ slash + "PC_aligned.tif", aligned_stack)

        top_stack, bottom_stack = cut_stack_top_and_bottom(aligned_stack)
        imsave(file_name + slash + 'top.tif', top_stack)
        imsave(file_name + slash + 'bottom.tif', bottom_stack)

        print("FOV " + fov + " aligned and saved.")

def align_subpixel_and_cut_experiment(path_file, list_fov, start_index_pic, end_index_pic):

    for fov in list_fov:

        file_name = path_file + slash + str(fov)

        aligned_stack = align_subpixel_one_fov_PC_only(file_name, start_index_pic, end_index_pic)
        imsave(file_name+ slash + "PC_aligned_spx.tif", aligned_stack)

        top_stack, bottom_stack = cut_stack_top_and_bottom(aligned_stack)
        imsave(file_name + slash + 'top_spx.tif', top_stack)
        imsave(file_name + slash + 'bottom_spx.tif', bottom_stack)

        print("FOV " + fov + " aligned and saved.")


def align_one_stack(stack_multi_array):

    new_stack_m = []

    start_inx, end_inx = 0, stack_multi_array.shape[0]

    #align al the subsequent pictures to that one using cross correlation
    image_array_reference = stack_multi_array[start_inx]

    for idx in range(start_inx+1, end_inx):

        image_array  = stack_multi_array[idx]


        shift_m = shift_scikit(image_array_reference, image_array)

        image_3 = registration.shift_image(image_array, shift_m, background='blank')
        new_stack_m.append(image_3)


    new_stack_m = np.array(new_stack_m)
    return new_stack_m

def cut_top_and_align_experiment(path_file, list_fov, position_to_align, start_index_pic, end_index_pic):


    for fov in list_fov:


        folder_image = path_file + slash + str(fov)
        name_image = folder_image + slash + "PC" + slash  + str(start_index_pic) + ".tif"

        t = TiffFile(name_image)
        image = t.asarray()
        t.close()

        image_0_cutted = cut_image_top(image)

        cutted_stack = []
        cutted_stack.append(image_0_cutted)

        for idx in range(start_index_pic + 1, end_index_pic):

            name_image = folder_image + slash + "PC" + slash + str(idx) + ".tif"

            t = TiffFile(name_image)
            image = t.asarray()
            t.close()

            image_cutted = cut_image_top(image)

            shift_m = shift_scikit(image_0_cutted, image_cutted)
            image_3 = registration.shift_image(image_cutted, shift_m, background='blank')
            cutted_stack.append(image_3)

        cutted_stack = np.array(cutted_stack)

        imsave(folder_image + slash + position_to_align + '.tif', cutted_stack)

        print("FOV " + fov + " position " + position_to_align + " cut, aligned and saved.")

def cut_and_align_experiment_version_1(path_file, list_fov, position_to_align, start_index_pic, end_index_pic):

    for fov in list_fov:

        pc_path_file = path_file + slash + str(fov) + slash + "PC"

        stack = create_stack(pc_path_file,  start_index_pic, end_index_pic)
        top_stack, bottom_stack = cut_stack_top_and_bottom(stack)

        aligned_stack = align_one_stack(top_stack)
        imsave(path_file + slash + str(fov) + slash + position_to_align + '.tif', aligned_stack)

        print("FOV " + fov + " position " + position_to_align+ " cut, aligned and saved.")

def align_experiment(path_file, list_fov, start_index_pic, end_index_pic):

    for fov in list_fov:

        file_name = path_file + slash + str(fov)

        aligned_stack = align_one_fov_PC_only(file_name, start_index_pic, end_index_pic)
        imsave(file_name+ slash + "PC_aligned.tif", aligned_stack)

        print("FOV " + fov + " aligned and saved.")

def cut_experiment(path_file, list_fov, start_index_pic, end_index_pic):

    for fov in list_fov:

        file_name = path_file + slash + str(fov)

        aligned_stack = file_name + slash + "PC_aligned.tif"

        top_stack, bottom_stack = cut_stack_top_and_bottom(aligned_stack)
        imsave(file_name + slash + 'top.tif', top_stack)
        imsave(file_name + slash + 'bottom.tif', bottom_stack)

        print("FOV " + fov + " aligned and saved.")

def cut_stack_top_and_bottom(t_array):

    ymin_top = t_array.shape[1] // 2-10
    ymax_top = ymin_top - 250
    xmin_top = 10
    xmax_top = t_array.shape[2] - 10

    with DebugPlot('graph') as p:
        p.title("top")
        p.imshow(image_reference.image, cmap='rainbow')
        p.ylim(ymin_top, ymax_top)
        p.xlim(xmin_top, xmax_top)

    # bottom -- hight and width to cut
    ymin_bottom = t_array.shape[1] // 2+10
    ymax_bottom = ymin_bottom + 250
    xmin_bottom = xmin_top
    xmax_bottom = xmax_top


    with DebugPlot('graph') as p:
        p.title("bottom")
        p.imshow(image_reference.image, cmap='rainbow')
        p.ylim(ymin_bottom, ymax_bottom)
        p.xlim(xmin_bottom, xmax_bottom)
        p.show()

    top_array_corrected = []
    bottom_array_corrected = []

    for t_array_i in t_array:
        top_image = t_array_i[ymax_top:ymin_top, xmin_top:xmax_top]
        top_array_corrected.append(top_image)

        bottom_image = t_array_i[ymin_bottom:ymax_bottom, xmin_bottom:xmax_bottom]
        bottom_array_corrected.append(bottom_image)

    top_array_corrected = np.array(top_array_corrected)

    bottom_array_corrected = np.array(bottom_array_corrected)
    return [top_array_corrected, bottom_array_corrected]

def cut_image_top(t_array):

    ymin_top = t_array.shape[0] // 2+20
    ymax_top = ymin_top - 250
    xmin_top = 10
    xmax_top = t_array.shape[1] - 10


    top_image = t_array[ymax_top:ymin_top, xmin_top:xmax_top]

    top_image = np.array(top_image)

    return top_image

def align_PC_and_FP_and_cut_experiment(path_file, list_fov, start_index_pic, end_index_pic, FP, rotate = False):

    for fov in list_fov:

        file_name = path_file + "/" + str(fov)

        if rotate:
            aligned_stack_pc, aligned_stack_fp = align_and_rotate_one_fov_PC_FP(file_name, start_index_pic, end_index_pic, FP)
        else:
            aligned_stack_pc, aligned_stack_fp = align_one_fov_PC_FP(file_name, start_index_pic, end_index_pic, FP)



        imsave(file_name+ "/PC_aligned.tif", aligned_stack_pc)
        imsave(file_name + "/" + FP + "_aligned.tif", aligned_stack_fp)

        top_stack, bottom_stack = cut_stack_top_and_bottom(aligned_stack_pc)
        imsave(file_name + '/top.tif', top_stack)
        imsave(file_name + '/bottom.tif', bottom_stack)

        top_stack, bottom_stack = cut_stack_top_and_bottom(aligned_stack_fp)
        imsave(file_name + '/top_' + FP.lower() + '.tif', top_stack)
        imsave(file_name + '/bottom_' + FP.lower() + '.tif', bottom_stack)

        print("FOV " + fov + " PC and " + FP + " - aligned and saved.")

def align_PC_and_RFP_and_cut_experiment(path_file, list_fov, start_index_pic, end_index_pic, rotate = False):

    for fov in list_fov:

        file_name = path_file + "/" + str(fov)

        if rotate:
            aligned_stack_pc, aligned_stack_rfp = align_and_rotate_one_fov_PC_RFP(file_name, start_index_pic, end_index_pic)
        else:
            aligned_stack_pc, aligned_stack_rfp = align_one_fov_PC_RFP(file_name, start_index_pic, end_index_pic)



        imsave(file_name + slash + "PC_aligned.tif", aligned_stack_pc)
        imsave(file_name + slash + "RFP_aligned.tif", aligned_stack_rfp)

        top_stack, bottom_stack = cut_stack_top_and_bottom(aligned_stack_pc)
        imsave(file_name + slash + 'top.tif', top_stack)
        imsave(file_name + slash + 'bottom.tif', bottom_stack)

        top_stack, bottom_stack = cut_stack_top_and_bottom(aligned_stack_rfp)
        imsave(file_name + slash + 'top_rfp.tif', top_stack)
        imsave(file_name + slash + 'bottom_rfp.tif', bottom_stack)

        print("FOV " + fov + " PC and RFP - aligned and saved.")

slash = "/";
