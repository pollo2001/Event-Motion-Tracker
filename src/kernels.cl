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

__kernel void generate_events(
  __global const uchar* current_frame,
  __global const uchar* previous_frame,
  __global char* event_map,
  const int threshold,
  const int width,
  const int height){

  // 2D threads
    int x = get_global_id(0); //row index (x goes up to height)
    int y = get_global_id(1); //col index (y goes up to width)

    //set bounds to width and height
    if (x < height && y < width){

      //flatten 2D coordinates to 1D array
      int idx = x * width + y;

      //read intensity at (x,y)
      int current_value = (int)current_frame[idx];
      int previous_value = (int)previous_frame[idx]; //save previous frame for historical data

      //compute delta
      int delta = current_value - previous_value;


      //delta cases to write to event map
      if (delta > threshold){
          event_map[idx] = +1; //if positive, on event (+1)
        }

      else if (delta < (-1)*threshold){
        event_map[idx] = -1; //if negative, off event (-1)
      }

      else {
        event_map[idx] = 0;   //if no change, no event (0)
      }
    }
  }






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

__kernel void generate_sae(
  __global char* event_map,
  __global float* surface,
  const float alpha,
  const int width,
  const int height){
  
   int x = get_global_id(0); //row index (x goes up to height)
   int y = get_global_id(1); //col index (y goes up to width)

     //set bounds to width and height
    if (x < height && y < width){

      //flatten 2D coordinates to 1D array
      int idx = x * width + y;

      //read surface intensity at (x,y)
      float prior_surface = surface[idx];
     
      //apply decay factor
      float decayed_value = prior_surface * alpha;

      //if an event occured, reset surface to 255
      if (event_map[idx] != 0){
        surface[idx] = 255;
      }

      else 
      surface[idx] = decayed_value;
    }
}


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

__kernel void generate_boxFilter(
  __global const float* surface,
  __global float* output_surface,
  const float noise_floor_threshold,
  const int width,
  const int height){
  
   int x = get_global_id(0); //row index (x goes up to height)
   int y = get_global_id(1); //col index (y goes up to width)

  //this current pixel
   int center_idx = x * width + y;

     //set bounds to width and height, with clearance of 3
    if (x >= 1 && x < height - 1 && y >= 1 && y < width - 1){

      float sum  = 0; 

      //local 3x3 window around (x,y)
      //loop through 3 rows of the box
      for (int row_offset = -1; row_offset <=1; row_offset++){

        //loop through the 3 columns of the box for each row
        for (int col_offset = -1; col_offset <= 1; col_offset++){

          // calculate neighbor relative index
          int neighbor_x = x + row_offset;
          int neighbor_y = y + col_offset;

          //flatten the neighbor 2D into a 1D array
          int neighbor_idx = neighbor_x * width + neighbor_y;

          //read pixel from surface using neighbor_idx and add to sum
          sum = sum + surface[neighbor_idx];
        }
      }
         //average it by 9 pixels per box
          float filtered_pixel = sum / 9; 
          
          if (filtered_pixel < noise_floor_threshold)
            output_surface[center_idx] = 0;
          else 
          output_surface[center_idx] = filtered_pixel;
    }
  }