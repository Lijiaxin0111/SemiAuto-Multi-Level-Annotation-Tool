### Entering the Annotation Interface

```
python interactive_demo.py  --video ./demo_data/make_glass.mp4  --workspace ./workspace/make_glass --num_objects 1 --gpu 0
```

Wait a moment for data and video to load.

The annotation interface will then be ready.



### **UI Overview and Shortcut Keys**

### **Main Interface:**



**Tip:** The automated annotation tool has limitations. It may struggle with objects that have unclear contours or are too small, requiring manual corrections with the brush and eraser tools.





### Basic Functions and Shortcuts:

- `target_obj_id`: Switch the annotation target. eg:  press the number key `1` to annotate the object with `obj_id = 1`.
- **Single-frame annotation mode switching:**
  - `A` key (default): Auto Mode
    - Left-click to select an object using automatic annotation.
    - Right-click to remove an object using automatic annotation.
  - `I` key: Brush Mode
  - `E` key: Erase Mode
    - In Brush Mode and Erase Mode:
      - Use `-` and `=` to adjust brush size.
      - Press `0` to enable magnifier mode.
      - Use `[` and `]` to adjust the magnifier size.
- **Multi-frame propagation:**
  - `F` key: Propagates the current mask and annotated frames forward automatically. Press again to pause.
  - `B` key: Propagates the current mask and annotated frames backward automatically. Press again to pause.

💡 **Annotation Tip:** First, annotate the mask for the initial frame. Then, use the `F` key to propagate one frame at a time, check for missing or incorrect areas, correct them in single-frame mode, and continue propagation frame by frame.

- **Uncertain Mask Annotation:**

  - Press `V` to enter Paint_VOID_MASK mode, allowing you to mark uncertain regions caused by motion blur or poor image quality. These pixels will be excluded from the final evaluation.
  - Press `X` to enter Erase_VOID_MASK mode, which erases previously marked VOID_MASK areas.

  **Tips:**

  - VOID_MASK appears as a light white overlay in the UI.
  - Press `A`, `I`, or `E` to return to Auto Mode, Brush Mode, or Erase Mode, respectively.

  **VOID_MASK  in M-cube-VOS:**

  - Use VOID_MASK for motion blur areas where contours are unclear. Example: ghosting trails of fast-moving objects.
  - Use VOID_MASK for small, indistinct object fragments. Example: tiny water droplets in splashes.
  - Use VOID_MASK for unclear transparent object contours. Example: the thick bottle walls of a transparent water bottle.
  - Use VOID_MASK when two objects blend together and their boundaries are indistinguishable. Example: mixed egg whites from broken eggs in the same bowl.

- **Color Difference Mask Mode:**

  - While in Paint_MASK, Erase_MASK, Paint_VOID_MASK, or Erase_VOID_MASK mode, press `T` to activate the color selection mode. Click on the target pixel in the image to enter **Color Difference Mask Mode**.
  - In this mode, scroll the mouse wheel **up/down** to increase/decrease color tolerance, adjusting the mask area.
  - Press `Y` to annotate the exposed pixels within the mask.
  - Press `T` again to exit Color Difference Mask Mode.

  **Usage Tips:**

  - Useful for scenes with distinct colors, such as smoke or mist.
  - Press `I` to enter Paint_MASK mode.
  - Press `T` to activate color selection mode.
  - Click on the target color to enter **Color Difference Mask Mode**.
  - Adjust tolerance carefully with the mouse wheel to fully cover the target object.
  - Press `Y` to annotate the selected pixels.
  - Press `T` to exit Color Difference Mask Mode.
  - Use Paint_MASK or Erase_MASK mode to refine the color-based mask.

- **Mask Visualization Color Adjustment:**

  ![viz_mask_adjust_UI](D:\LabResearch\projects\DeformVOS\open_code\fig\viz_mask_adjust_UI.png)

  - Click the **color block on the right side** of the lower-left UI panel to open the color picker. Select a color and click OK to apply.

### **FAQ:**

**Q: If** `target_obj` **contains multiple objects (e.g., obj_1, obj_2, obj_3), how can I annotate multiple objects?**

1. Modify the command-line parameters:

   Add `--num_objects` specifying the number of objects to annotate. Example: If annotating three objects, use:

   ```
   --num_objects 3
   ```

2. Switch objects in the annotation interface:

   - Use the **target_object selection box** in the lower-left corner to choose the current annotation target.
   - Alternatively, press number keys `1`, `2`, `3`, etc., to switch between objects.

3. Once switched, use **Auto Mode, Paint Mode, Erase Mode, or Color Mask Mode** to annotate each object’s mask accordingly.