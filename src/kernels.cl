//difference, decay, filter

//TO--DO
/**
LAYER 1:
Generate Events from Frame Difference
1. Fetch current 2D pixel index (x, y). Guard against out-of-bounds.
2. Read intensity at (x, y) from current_frame and previous_frame.
3. Compute signed change: delta = current - previous.
4. If delta > +threshold:
      Write +1 (ON event: pixel got brighter) to event_map.
    Else if delta < -threshold:
      Write -1 (OFF event: pixel got darker) to event_map.
    Else:
      Write  0 (No event) to event_map.
**/


/**
LAYER 2:
Decaying Time Surface (SAE)
1. Fetch current 2D pixel index (x, y). Guard against out-of-bounds.
2. Read prior surface intensity at (x, y).
3. Apply decay factor: decayed_value = prior_surface * alpha  (e.g., alpha = 0.85).
4. Read event polarity from Layer 1 at (x, y):
      If an event occurred (polarity != 0):
          Reset surface at (x, y) to maximum activity (e.g., 255.0).
      Else:
          Store decayed_value at (x, y).
**/


/**
LAYER 3:
Spatial Box Filter - remove noise before centroid extraction
1. Fetch current 2D pixel index (x, y). Guard boundary edges (1 to width-2, 1 to height-2).
2. Iterate through local 3x3 window around (x, y).
3. Sum neighbor intensities from Layer 2 decaying surface.
4. Compute average: filtered_pixel = sum / 9.0.
5. If filtered_pixel < noise_floor_threshold:
      Write 0.0 (suppress faint background noise).
    Else:
      Write filtered_pixel to output_surface.
**/

