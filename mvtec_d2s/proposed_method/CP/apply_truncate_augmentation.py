from .lib import *
from .blending_masks import build_blending_regions


# ============================================================
# TRUNCATION AUGMENTATION
# ============================================================

def apply_truncate_augmentation(
        object_extracted, contour_points, x, y, w, h,
        background_width, background_height, kernel_size, crop_direction,
        rng=None, use_blending=True):

    if rng is None:
        rng = random

    #crop_direction = 'right'

    # ========================================================
    # TOP
    # ========================================================

    if crop_direction == 'top':

        hh = rng.uniform(0.1*h, 0.5 * h)
        object_extracted = object_extracted[y + int(hh):, :]

        y_offset = 0
        x_offset = rng.randint(0, abs(background_width - (w + (2*kernel_size))))
    

        contour_points[:, 1] -= int(hh) + y
        
        contour_points[:, 0] += 2*kernel_size

        
        xn, yn, wn, hn = cv2.boundingRect(contour_points)

        mask_image_gray = cv2.cvtColor(object_extracted, cv2.COLOR_BGR2GRAY)
        _, binary_mask_single_channel = cv2.threshold(mask_image_gray, 1, 255, cv2.THRESH_BINARY)
        
        padded_mask = np.pad(
            binary_mask_single_channel,
            pad_width=((0, 2 * kernel_size), (2 * kernel_size, 2 * kernel_size)),  # Bottom padding on height (y) and both sides on width (x)
            mode='constant',
            constant_values=0  # Black padding
        )

        
    
        crop_slices = (
            slice(0, yn + hn + kernel_size),
            slice(xn - kernel_size, xn - kernel_size + wn + (2 * kernel_size)),
        )
        (
            binary_mask_single_channel_crop,
            blurred_edge_region,
            outer_region,
            inner_region,
        ) = build_blending_regions(
            padded_mask,
            crop_slices,
            kernel_size,
            use_blending=use_blending,
        )
        
        
        object_extracted_padded = np.pad(
            object_extracted,
            pad_width=((0, 2 * kernel_size), (2 * kernel_size, 2 * kernel_size), (0, 0)),  # Bottom padding on height (y), both sides on width (x), no padding on channels
            mode='constant',
            constant_values=0  # Black padding (0,0,0 in RGB)
        )


    
        mask_image = object_extracted_padded[0:yn+hn+kernel_size, xn-kernel_size:xn-kernel_size+wn+(2*kernel_size)]

        #cv2.imwrite("binary_mask_single_channel_crop.jpg",binary_mask_single_channel_crop)

        #cv2.imwrite('mask_blur.png', mask_blur)

        #cv2.imwrite('outer_region.png', outer_region )

        #cv2.imwrite('blurred_edge_region.png', blurred_edge_region)

        #cv2.imwrite('mask_image.png', mask_image )
        #cv2.imwrite('inner_region.png', inner_region )

        x_min, y_min = np.min(contour_points, axis=0)

        contour_points[:, 0] -= x_min - kernel_size 
        contour_points[:, 1] -= y_min - yn 

    # ========================================================
    # LEFT
    # ========================================================

    elif crop_direction == 'left':

        ww = rng.uniform(0.1 * w, 0.5 * w)
        object_extracted = object_extracted[:, int(ww) + x:]
        y_offset = rng.randint(0, abs(background_height - (h + (2*kernel_size))))
        x_offset = 0
        contour_points[:, 0] -= int(ww) + x
        
        contour_points[:, 1] += 2*kernel_size

        xn, yn, wn, hn = cv2.boundingRect(contour_points)

        mask_image_gray = cv2.cvtColor(object_extracted, cv2.COLOR_BGR2GRAY)
        _, binary_mask_single_channel = cv2.threshold(mask_image_gray, 1, 255, cv2.THRESH_BINARY)

        padded_mask = np.pad(
            binary_mask_single_channel,
            pad_width=((2 * kernel_size, 2 * kernel_size), (0, 2 * kernel_size)),  # Padding on height (y) and only on the right side of width (x)
            mode='constant',
            constant_values=0  # Black padding
        )

        
        
        crop_slices = (
            slice(yn - kernel_size, yn - kernel_size + hn + (2 * kernel_size)),
            slice(0, xn + wn + kernel_size),
        )
        (
            binary_mask_single_channel_crop,
            blurred_edge_region,
            outer_region,
            inner_region,
        ) = build_blending_regions(
            padded_mask,
            crop_slices,
            kernel_size,
            use_blending=use_blending,
        )
        
        
        object_extracted_padded = np.pad(
            object_extracted,
            pad_width=((2 * kernel_size, 2 * kernel_size), (0, 2 * kernel_size), (0, 0)),  # Padding on height (y) and only on the right side of width (x), no padding on channels
            mode='constant',
            constant_values=0  # Black padding (0,0,0 in RGB)
        )


        mask_image = object_extracted_padded[yn-kernel_size:yn-kernel_size+hn+(2*kernel_size), 0:xn+wn+kernel_size]

        x_min, y_min = np.min(contour_points, axis=0)

        contour_points[:, 0] -= x_min - xn  
        contour_points[:, 1] -= y_min - kernel_size

    # ========================================================
    # BOTTOM
    # ========================================================

    elif crop_direction == 'bottom':

        hh = rng.uniform(0.5 * h , 0.9 * h)
        object_extracted = object_extracted[:int(hh) + y, :]
        
        #cv2.imwrite("object_extracted.jpg",object_extracted)


        y_offset = background_height - (int(hh) +  kernel_size)
        x_offset = rng.randint(0, abs(background_width - (w + (2*kernel_size))))
        
        
        contour_points += 2*kernel_size

        #print(object_extracted.shape)

        #contour_points[:, 1] += int(hh) + y
        #print(x)
        #print(w)


        xn, yn, wn, hn = cv2.boundingRect(contour_points)
        #print(xn)
        #print(wn)

        mask_image_gray = cv2.cvtColor(object_extracted, cv2.COLOR_BGR2GRAY)
        _, binary_mask_single_channel = cv2.threshold(mask_image_gray, 1, 255, cv2.THRESH_BINARY)

        padded_mask = np.pad(
            binary_mask_single_channel,
            pad_width=((2 * kernel_size, 0), (2 * kernel_size, 2 * kernel_size)),  # Padding on top of height (y) and both sides of width (x)
            mode='constant',
            constant_values=0  # Black padding
        )
        
        #print(padded_mask.shape)
        
        
        #cv2.imwrite("padded_mask.jpg",padded_mask)

        
        crop_slices = (
            slice(yn - kernel_size, int(hh) + yn),
            slice(xn - kernel_size, xn - kernel_size + wn + (2 * kernel_size)),
        )
        (
            binary_mask_single_channel_crop,
            blurred_edge_region,
            outer_region,
            inner_region,
        ) = build_blending_regions(
            padded_mask,
            crop_slices,
            kernel_size,
            use_blending=use_blending,
        )

        #print("binary_mask_single_channel_crop shape ", binary_mask_single_channel_crop.shape)
                            
        object_extracted_padded = np.pad(
            object_extracted,
            pad_width=((2 * kernel_size, 0), (2 * kernel_size, 2 * kernel_size), (0, 0)),  # Padding on top of height (y), both sides of width (x), no padding on channels
            mode='constant',
            constant_values=0  # Black padding (0,0,0 in RGB)
        )


        mask_image = object_extracted_padded[yn-kernel_size : int(hh)+yn, xn-kernel_size:xn-kernel_size+wn+(2*kernel_size)]
        

        
        x_min, y_min = np.min(contour_points, axis=0)

            

        contour_points[:, 0] -= x_min - kernel_size
        contour_points[:, 1] -= y_min - kernel_size  

    # ========================================================
    # RIGHT
    # ========================================================

    elif crop_direction == 'right':

        ww = rng.uniform(0.5 * w, 0.9 * w)
        object_extracted = object_extracted[:, :int(ww) + x]
        
        y_offset = rng.randint(0, abs(background_height - (h + (2*kernel_size))))
        x_offset = background_width - (int(ww) + kernel_size)
        
        contour_points += 2*kernel_size

        #x_offset = background_width - int(ww) 

        xn, yn, wn, hn = cv2.boundingRect(contour_points)

        mask_image_gray = cv2.cvtColor(object_extracted, cv2.COLOR_BGR2GRAY)
        _, binary_mask_single_channel = cv2.threshold(mask_image_gray, 1, 255, cv2.THRESH_BINARY)

        
        padded_mask = np.pad(
            binary_mask_single_channel,
            pad_width=((2 * kernel_size, 2 * kernel_size), (2 * kernel_size, 0)),  # Padding on height (y) and only on the left side of width (x)
            mode='constant',
            constant_values=0  # Black padding
        )

        crop_slices = (
            slice(yn - kernel_size, yn - kernel_size + hn + (2 * kernel_size)),
            slice(xn - kernel_size, int(ww) + xn),
        )
        (
            binary_mask_single_channel_crop,
            blurred_edge_region,
            outer_region,
            inner_region,
        ) = build_blending_regions(
            padded_mask,
            crop_slices,
            kernel_size,
            use_blending=use_blending,
        )


        object_extracted_padded = np.pad(
            object_extracted,
            pad_width=((2 * kernel_size, 2 * kernel_size), (2 * kernel_size, 0), (0, 0)),  # Padding on height (y), only on the left side of width (x), no padding on channels
            mode='constant',
            constant_values=0  # Black padding (0,0,0 in RGB)
        )


        mask_image = object_extracted_padded[yn-kernel_size:yn-kernel_size+hn+(2*kernel_size), xn-kernel_size:int(ww)+xn ]

        #cv2.imwrite("binary_mask_single_channel_crop.jpg",binary_mask_single_channel_crop)

        #cv2.imwrite('mask_blur.png', mask_blur)

        #cv2.imwrite('outer_region.png', outer_region )

        #cv2.imwrite('blurred_edge_region.png', blurred_edge_region)

        #cv2.imwrite('mask_image.png', mask_image )
        #cv2.imwrite('inner_region.png', inner_region )
                            
        x_min, y_min = np.min(contour_points, axis=0)
                    
        contour_points[:, 0] -= x_min - kernel_size
        contour_points[:, 1] -= y_min - kernel_size  





    return (
        contour_points,
        mask_image,
        binary_mask_single_channel_crop,
        blurred_edge_region,
        outer_region,
        inner_region,
        x_offset,
        y_offset
    )

