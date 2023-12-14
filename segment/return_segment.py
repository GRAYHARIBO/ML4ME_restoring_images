import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.segment.general import scale_masks

class segment_image(object):

    def __init__(self):
        self.mask_number = None

    def extract_segmentation_array(self, im0, im, masks, save_dir, bbox):
        max_detection = 15
        mask_array_cropped = np.moveaxis(masks.numpy(), 0, -1)
        num_mask = mask_array_cropped.shape[2]
        Nx = im0.shape[0]; Ny = im0.shape[1]
        mask_array = scale_masks(im.shape[2:], mask_array_cropped, [Nx, Ny, num_mask])
        mask_array_concat = np.zeros([Nx, Ny])
        for n in range(num_mask):
            mask_array_concat = mask_array_concat + np.squeeze(mask_array[:,:,n])*(n+1)
        
        cmap = mpl.colors.ListedColormap(['dimgray', 'red', 'orange', 'green', 'aqua', 'blue', 'purple', 
                                          'pink', 'chocolate', 'yellow', 'lime', 'skyblue', 'navy', 'magenta', 
                                          'saddlebrown', 'olive', 'darkseagreen', 'cornflowerblue', 'indigo', 'hotpink', 'white'])
        norm = mpl.colors.BoundaryNorm(np.arange(-0.5, max_detection+1, 1), cmap.N)
        labels = ['Mask {}'.format(i+1) for i in range(num_mask)]
        labels.insert(0, 'N/A')
        for n in range(max_detection-num_mask):
            labels.append('N/A')
           
        fig0, ax = plt.subplots(1, 1, figsize = (4, 6))
        ax.imshow(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB), zorder = 1)
        im = ax.imshow(mask_array_concat, cmap = cmap, norm = norm, alpha = 0.7, zorder = 2)
        ax.set_title('Segmentation Array', fontsize = 8)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig0.colorbar(im, ax = ax, shrink = 0.7)
        cbar.ax.set_yticks(np.arange(0, max_detection+1, 1))
        cbar.ax.set_yticklabels(labels, fontsize = 8)
        #fig0.savefig(os.path.join(save_dir, 'segmentation.jpg'))

        # GUI for Mask Selection
        root = Tk()
        root.geometry('800x600')
        # Segmented Figure Display
        frame1 = Frame(root, relief = 'solid', bd = 2)
        frame1.pack(side = 'left', fill = 'both', expand = True)
        canvas = FigureCanvasTkAgg(fig0, master = frame1)
        canvas.get_tk_widget().pack(side = TOP, fill = BOTH)
        canvas.draw()
        # Buttons for Mask Selection
        frame2 = Frame(root, relief = 'solid', bd = 2)
        frame2.pack(side = 'right', fill = 'both', expand = True)
        for i in range(num_mask):
            def button(i = i):
                self.mask_number = i
                print('Mask {} Selected'.format(i+1))
                root.destroy()
            button = Button(frame2, text = 'Mask {}'.format(i+1), font = 8,
                            overrelief='solid', width = 7, height = 2,
                            highlightcolor = 'dimgray', 
                            command = button,
                            repeatdelay=1000, repeatinterval=100)
            button.pack(anchor = 'center', pady = 3)
        root.mainloop()

        MASK_ARRAY = np.where(mask_array_concat == self.mask_number + 1, True, False)
        resized_mask = cv2.blur(mask_array[:,:,self.mask_number], (41,41))
        RESIZED_MASK = np.where(resized_mask > 0, True, False)
        BBOX = np.squeeze(bbox[self.mask_number, :])

        # fig1, ax = plt.subplots(1, 1, figsize = (20, 30))
        # ax.imshow(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB), zorder = 1)
        # im = ax.imshow(RESIZED_MASK, alpha = 0.7, zorder = 2)
        # ax.set_title('Resized Array', fontsize = 8)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # cbar = fig1.colorbar(im, ax = ax, shrink = 0.7)
        # cbar.ax.set_yticks(np.arange(0, max_detection+1, 1))
        # cbar.ax.set_yticklabels(labels, fontsize = 8)
        #fig1.savefig(os.path.join(save_dir, 'resized.jpg'))

        return MASK_ARRAY, RESIZED_MASK, BBOX
    

class segment_video(object):

    def __init__(self):
        self.mask_number = None
        self.updated_mask_number = None

    def extract_segmentation_array_video(self, im0, im, masks, save_dir, bbox, bbox_tracked):
        max_detection = 15
        mask_array_cropped = np.moveaxis(masks.numpy(), 0, -1)
        num_mask = mask_array_cropped.shape[2]
        num_mask_tracked = bbox_tracked.shape[0]
        Nx = im0.shape[0]; Ny = im0.shape[1]
        mask_array = scale_masks(im.shape[2:], mask_array_cropped, [Nx, Ny, num_mask])

        mask_array_concat = np.zeros([Nx, Ny])
        for n in range(num_mask):
            for m in range(num_mask_tracked):
                centerbbox = np.array(bbox[n,:])
                centerbbox_tracked = np.array(bbox_tracked[m,:4])
                bbox_dif = np.mean(np.abs(centerbbox - centerbbox_tracked))
                if bbox_dif < 2:
                    mask_array_concat = mask_array_concat + np.squeeze(mask_array[:,:,n]*bbox_tracked[m,8])
       
        num_mask = int(np.max(mask_array_concat))
        
        if self.mask_number == None or self.mask_number + 1 != self.updated_mask_number:
            cmap = mpl.colors.ListedColormap(['dimgray', 'red', 'orange', 'green', 'aqua', 'blue', 'purple', 
                                        'pink', 'chocolate', 'yellow', 'lime', 'skyblue', 'navy', 'magenta', 
                                        'saddlebrown', 'olive', 'darkseagreen', 'cornflowerblue', 'indigo', 'hotpink', 'white'])
            norm = mpl.colors.BoundaryNorm(np.arange(-0.5, max_detection+1, 1), cmap.N)
            labels = ['Mask {}'.format(i+1) for i in range(num_mask)]
            labels.insert(0, 'N/A')
            for n in range(max_detection-num_mask):
                labels.append('N/A')
            
            fig0, ax = plt.subplots(1, 1, figsize = (4, 6))
            ax.imshow(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB), zorder = 1)
            im = ax.imshow(mask_array_concat, cmap = cmap, norm = norm, alpha = 0.7, zorder = 2)
            ax.set_title('Segmentation Array', fontsize = 8)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig0.colorbar(im, ax = ax, shrink = 0.7)
            cbar.ax.set_yticks(np.arange(0, max_detection+1, 1))
            cbar.ax.set_yticklabels(labels, fontsize = 8)

            # GUI for Mask Selection
            root = Tk()
            root.geometry('800x600')
            # Segmented Figure Display
            frame1 = Frame(root, relief = 'solid', bd = 2)
            frame1.pack(side = 'left', fill = 'both', expand = True)
            canvas = FigureCanvasTkAgg(fig0, master = frame1)
            canvas.get_tk_widget().pack(side = TOP, fill = BOTH)
            canvas.draw()
            # Buttons for Mask Selection
            frame2 = Frame(root, relief = 'solid', bd = 2)
            frame2.pack(side = 'right', fill = 'both', expand = True)
            for i in range(num_mask):
                def button(i = i):
                    self.mask_number = i
                    print('Mask {} Selected'.format(i+1))
                    root.destroy()
                button = Button(frame2, text = 'Mask {}'.format(i+1), font = 8,
                                overrelief='solid', width = 7, height = 2,
                                highlightcolor = 'dimgray', 
                                command = button,
                                repeatdelay=1000, repeatinterval=100)
                button.pack(anchor = 'center', pady = 3)
            root.mainloop()
            self.updated_mask_number = self.mask_number + 1

        MASK_ARRAY = np.where(mask_array_concat == self.mask_number + 1, True, False)
        resized_mask = cv2.blur(mask_array[:,:,self.mask_number], (41,41))
        RESIZED_MASK = np.where(resized_mask > 0, True, False)

        return MASK_ARRAY, RESIZED_MASK







