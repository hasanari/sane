function Frame(fname, data) {
	this.fname = fname;
	// this.pointcloud = null;
	this.data = data;
	this.last_bbox_id = 0;
	this.bounding_boxes = [];
	this.ys = [];
	this.evaluator = new Evaluator();
	this.annotated = false;
	this.mask_rcnn_indices = [];

	var k = 0;
    for ( var i = 0, l = this.data.length / DATA_STRIDE; i < l; i ++ ) {
  
        this.ys.push(this.data[ DATA_STRIDE * k + 2 ]);
        
        k++;
    }

	this.output = function() {
		return new OutputFrame(this);
	};

	this.scene_add_frame_children = function() {
		for (var i = 0; i < this.bounding_boxes.length; i++) {
				var box = this.bounding_boxes[i];
				scene.add(box.points);
            	scene.add(box.boxHelper);
                if(box.bbox3d_helper){
                    scene.add(box.bbox3d_helper);
                }
                box.changeBoundingBoxColor(default_color);
            	container.appendChild(box.text_label.element);
			}
	};

	this.scene_add_frame_bounding_box = function() {
		for (var i = 0; i < this.bounding_boxes.length; i++) {
			var box = this.bounding_boxes[i];
        	scene.add(box.boxHelper);
        	box.changeBoundingBoxColor(COLOR_WHITE);
        	if (box.text_label) {
        		container.appendChild(box.text_label.element);
        	}
		}
	};

	this.scene_remove_frame_children = function() {
		for (var i = 0; i < this.bounding_boxes.length; i++) {
				var box = this.bounding_boxes[i];
				scene.remove(box.points);
            	scene.remove(box.boxHelper);
            	box.text_label.element.remove();
			}
        
        /*
        for (var i = scene.children.length - 1; i >= 0; i--) {

            //console.log("scene3.children[i].geometry ", scene3.children[i] );
            
            if(scene.children[i]){
                
              
                     
                if( scene.children[i].geometry &&  typeof scene.children[i].geometry.dispose === 'function' ){
                   
                    if( scene.children[i].material.color.r == 1 && scene.children[i].material.color.g == 1 && scene.children[i].material.color.b == 1){
                         console.log(scene.children[i]);
                    
                        if(scene.children[i].text_label && scene.children[i].text_label.element){
                        
                                scene.children[i].text_label.element.remove();

                                scene.children[i].geometry.dispose();            

                                scene.children[i].material.dispose();     
                            
                                scene.remove(scene.children[i]);
                            delete scene.children[i];
                        }
                       
                        
                    }
                }
                
                
            }
            

            //scene3.children[i] = undefined;//or

            
        }
        */
        
	};

	this.is_annotated = function() {
		return this.bounding_boxes.length > 0;
	};

}

function OutputFrame(frame) {
	this.fname = frame.fname;
	this.bounding_boxes = [];
	// this.evaluator = new OutputEvaluator(frame.evaluator);

	for (var i = 0; i < frame.bounding_boxes.length; i++) {
		this.bounding_boxes.push(frame.bounding_boxes[i].output());
	}
}
	
