/* Class for handling file reading/writing, bounding box drawing */
function App() {
    this.mmax;
    this.mmin;
    this.mcenter;
    this.annotatorID='Guest';
    this.selectedBoxWidth=0;
    this.selectedBox;
    this.selectedBox3DView;
    this.forceVisualize = false;
    this.isRedColor = true;
    this.isLiveRecolors = false;
    this.annote_pointcloudXZ;
    this.annote_pointcloudXZY;
    this.masked_indices;
    this.bboxObject;
    this.opt_box;
    this.fnames = [];
    this.tempBBOX = [];
    this.not_update_all_bbox = false;
    this.frames = {};
    this.prev_viz = {
        r: 0.0,
        w: 0.0,
        l: 0.0,
        h: 0.0,
        a: 0.0,
        x: 0.0,
        y: 0.0,
        z: 0.0
    };
    this.cur_frame;
    this.isAutoDrawOn =false;
    this.cur_pointcloud;
    this.move2D = false;
    this.eps = 0.00001;
    this.show_prev_frame;
    this.editing_box_id;
    this.evaluators = [];
    this.controls = {};
    this.lock_frame = false;

    this.init = function() {
        $.ajax({
            context: this,
            url: '/loadFrameNames',
            type: 'POST',
            contentType: 'application/json;charset=UTF-8',
            success: function(response) {
                this.drives = parsePythonJSON(response);
                var drive_keys = Object.keys(this.drives);
                drive_keys.sort();
                for (var i = 0; i < drive_keys.length; i++) {
                    var drive = drive_keys[i];
                    for (var j = 0; j < this.drives[drive].length; j++) {
                        
                        if(j > 1000){
                            break;
                        }
                        var fname = pathJoin([drive, this.drives[drive][j].split('.')[0]]);
                        this.fnames.push(fname);
                        addFrameRow(fname);
                        this.controls[fname] = i;
                    }
                }
                this.set_frame(this.fnames[0]);
                focus_frame_row(getFrameRow(this.fnames[0]));
            },
            error: function(error) {
                console.log(error);
            }
        });
    };

    this.get_prev_fname = function(fname) {
        var idx = this.fnames.indexOf(fname);
        if (idx == 0) {
            return ""
        }
        return this.fnames[idx - 1];
    }

    this.get_frame = function(fname) {
        if (fname in this.frames) {
            return this.frames[fname];
        } else {
            return false;
        }
    };
    
    this.get_frame_idx= function(fname) {
        
        var idx = this.fnames.indexOf(fname);
        return idx;
    };


    this.bbox_visualization_clearance = function() {

        $("#footer-top-view").hide();
        $("#panel3").hide();
        $("#panel").hide();
        $("#panel2").hide();

        for (var i = scene2.children.length - 1; i >= 0; i--) {

            scene2.remove(scene2.children[i]);
            delete scene2.children[i];
        }

        for (var i = scene3.children.length - 1; i >= 0; i--) {

            scene3.remove(scene3.children[i]);
            delete scene3.children[i];
        }

        if (this.cur_frame) {
            for (var i = 0; i < this.cur_frame.bounding_boxes.length; i++) {

                this.cur_frame.bounding_boxes[i].changeBoundingBoxColor(0xffff00);
            }
        }

        animate2();

    };

    this.bbox_visualization = function() {


        $("#footer-top-view").hide();
        
        
        if (selectedBox) {

            if (this.cur_frame) {
                for (var i = 0; i < this.cur_frame.bounding_boxes.length; i++) {

                    this.cur_frame.bounding_boxes[i].changeBoundingBoxColor(0xffff00);
                }
            }
            selectedBox.changeBoundingBoxColor(hover_color.clone());

            viz_start = (Date.now() / 1000);

            var points = this.cur_frame.data;

            var opt_box = new OutputBox(selectedBox);
            
            $("#summary-object-islocked").prop('checked', opt_box['islocked']);
            
            if( opt_box['islocked']){
                $("#summary-object-islocked").parent().css("color", "red");
            }else{
                $("#summary-object-islocked").parent().css("color", "white");
            }

            if (
                this.forceVisualize == false && 
                this.prev_viz.r == settingsControls.NeighborsRadius &&
                this.prev_viz.w == opt_box['width'] &&
                this.prev_viz.l == opt_box['length'] &&
                this.prev_viz.x == opt_box['center'].x &&
                this.prev_viz.y == opt_box['center'].y &&
                this.prev_viz.a == opt_box['angle'] &&
                this.prev_viz.islocked == opt_box['islocked']
            ) {

                return null;
            }

            $("#footer-top-view").show();



            $("#panel3").hide();
            $("#panel").hide();
            $("#panel2").hide();
            this.prev_viz.r = settingsControls.NeighborsRadius;
            this.prev_viz.w = opt_box['width'];
            this.prev_viz.l = opt_box['length'];
            this.prev_viz.x = opt_box['center'].x;
            this.prev_viz.y = opt_box['center'].y;
            this.prev_viz.a = opt_box['angle'];
            this.prev_viz.islocked = opt_box['islocked'];
            


            var _rest = get_point_inside_box(points, opt_box, app.cur_frame.mask_rcnn_indices, settingsControls.NeighborsRadius);
            var data = _rest[0];
            var py = _rest[1];
            var masked_indices = _rest[2];
            var annote_pointcloud = generateNewPointCloud(data, COLOR_RED, false);

            for (var j = 0; j < masked_indices.length; j++) {
                annote_pointcloud.geometry.colors[masked_indices[j]] = new THREE.Color(0x00ff6b);

            }
            annote_pointcloud.geometry.colorsNeedUpdate = true;

            annote_pointcloud.material.size =  settingsControls.PointSize+0.5;  // 1.0;


            var annote_pointcloudXZ = generateNewPointCloud(data, COLOR_RED, false);

            for (var j = 0; j < masked_indices.length; j++) {
                annote_pointcloudXZ.geometry.colors[masked_indices[j]] = new THREE.Color(0x00ff6b);
            }
            annote_pointcloudXZ.geometry.colorsNeedUpdate = true;
            annote_pointcloudXZ.material.size = annote_pointcloud.material.size;


            var count = 0;
            var colors = annote_pointcloudXZ.geometry.colors;
            
            var max_y = 0;
            var min_y = 99;


            for (var i = 0; i < annote_pointcloudXZ.geometry.vertices.length; i++) {
                var v = annote_pointcloudXZ.geometry.vertices[i];
                
                max_y = Math.max(max_y, v.y);
                min_y = Math.min(min_y, v.y);
                
                if (colors[i].b > colors[i].r) {
                    count += 1;
                    v.y = -0.001;
                } else {
                    v.y = 0;
                }
            }
            
            if( py.length > 10){            
                var max_y = Math.max(...py);
                var min_y = Math.min(...py);
            }

            //console.log(max_y, min_y);
            
            annote_pointcloudXZ.geometry.verticesNeedUpdate = true;


            this.annote_pointcloudXZ = annote_pointcloudXZ;
            this.annote_pointcloudXZY = annote_pointcloud;
            this.masked_indices = masked_indices;
            var center = selectedBox.get_center();

            var bbox_max = selectedBox.geometry.vertices[0].clone();
            var bbox_min = selectedBox.geometry.vertices[1].clone();


            bbox_max.y = max_y; // Math.max(...py)
            bbox_max.z = bbox_max.z - center.x;
            bbox_max.x = bbox_max.x - center.y;

            bbox_min.y = min_y- 0.2; // Value normalized from Denoise PointCNN  GROUND_TO_Z = 0.2
            bbox_min.z = bbox_min.z - center.x;
            bbox_min.x = bbox_min.x - center.y;

     

            if (selectedBox.height == 0) {

                var car_height = Math.min(Math.abs(bbox_max.y - bbox_min.y), 10.0);
                selectedBox.height = car_height;
            } else {
                car_height = selectedBox.height;
            }




            var bbox = new THREE.Box3();
            bbox.setFromCenterAndSize(new THREE.Vector3(0, bbox_min.y + (car_height / 2) , 0), new THREE.Vector3(opt_box.length, car_height, opt_box.width));

            
            var mat4 = new THREE.Matrix4();
            mat4.extractRotation(annote_pointcloud.matrixWorld);
            bbox.applyMatrix4( mat4 );
            var helper = new THREE.Box3Helper(bbox, 0xD4AF37);

            helper.rotation.y = opt_box.angle;

            this.bboxObject = bbox;
            this.bboxObjecthelper = helper;

            for (var i = scene2.children.length - 1; i >= 0; i--) {
                scene2.remove(scene2.children[i]);
                delete scene2.children[i];
            }


            
            if(false && opt_box['islocked'] == false){  
                var bbox3DotGeometry = new THREE.Geometry();

                this.bbox3_min = new THREE.Vector3(this.bboxObject.min.x, bbox_min.y, this.bboxObject.min.z);
                this.bbox3_max = new THREE.Vector3(this.bboxObject.min.x, bbox_min.y + car_height, this.bboxObject.min.z);

                
                bbox3DotGeometry.vertices.push(this.bbox3_max);
                bbox3DotGeometry.vertices.push( this.bbox3_min );

                //bbox3DotGeometry.vertices.push(new THREE.Vector3(0, bbox_min.y + car_height / 2, 0));

                var dotMaterial = new THREE.PointsMaterial({
                    size: 8,
                    sizeAttenuation: false
                });
                dotMaterial.color = COLOR_RED.clone();
                var z_dot = new THREE.Points(bbox3DotGeometry, dotMaterial);
                scene2.add(z_dot);

            }
            
            scene2.add(annote_pointcloud);
            scene2.add(helper);

            $("#summary-object-id").text("["+opt_box['box_id']+"]");
            $("#summary-object-type").text(opt_box['object_id']);
            $("#summary-object-dimension").text( parseFloat(opt_box['width']).toFixed(2) +" x "+ parseFloat(opt_box['length']).toFixed(2) +" x "+ parseFloat(car_height).toFixed(2) );
            $("#summary-object-angle").text( (parseFloat( (opt_box['angle']) * (180/Math.PI) ) % 360 ) .toFixed(2) + " degree");
            
            $("#summary-object-center-location").text("("+ parseFloat(opt_box['center'].x).toFixed(2) +", "+ parseFloat(opt_box['center'].y).toFixed(2)+")");
            
            // $("#summary-object-isvalidated").text("-");
            if(opt_box['is_auto_generated']){
            
                $("#summary-object-isautogenrated").text("Yes");
            }else{
            
                $("#summary-object-isautogenrated").text("No");
            }
            
             $("#panel2").show();

            

            if (this.not_update_all_bbox == false) { // Updated all views!!

                for (var i = scene3.children.length - 1; i >= 0; i--) {
                    scene3.remove(scene3.children[i]);
                    delete scene3.children[i];
                }

                camera3.position.set(0, bbox_max.z + 3, bbox_min.z - 3);

                controls3.enabled = false;
                controls3.update();

                scene3.add(annote_pointcloudXZ);

                opt_box['center'].x = 0.0;
                opt_box['center'].y = 0.0;

                bounding_boxes_selected = create_box_from_annot(opt_box, 0.0);

                if(bounding_boxes_selected == null ){
                    return;
                }

                if(opt_box['islocked'] == false){                    
                    scene3.add(bounding_boxes_selected.points);
                }

                bounding_boxes_selected.boxHelper.rotation.y = opt_box.angle;
                scene3.add(bounding_boxes_selected.boxHelper);




                var dotGeometry = new THREE.Geometry();
                dotGeometry.vertices.push(new THREE.Vector3(0, 0, 0));
                var dotMaterial = new THREE.PointsMaterial({
                    size: 5,
                    sizeAttenuation: false
                });
                var dot = new THREE.Points(dotGeometry, dotMaterial);
                scene3.add(dot);



                var size = 500;
                var divisions = 500;


                if (settingsControls["ShowGrid"]) {


                    var grid_helper = new THREE.GridHelper(size, divisions);
                    grid_helper.name = "grid_helper"
                    scene3.add(grid_helper);
                }

                this.selectedBox = bounding_boxes_selected;
                
                
                animate2();
                camera3.rotateZ(3.14);
            }else{
                
                animate2();
            }

            
            var __width = Math.max(opt_box.width, opt_box.length);
            
            if( this.selectedBoxWidth != Math.round(__width/2 ) ){
            
                this.selectedBoxWidth = Math.round(__width/2 );

                camera3.fov = 45 + 4 * (Math.abs(__width) / 2);
                camera2.fov = camera3.fov+5;

                camera3.updateProjectionMatrix();
                camera2.updateProjectionMatrix();
                
            }
            /*

            if( ( camera3.position.y <  6  ) &&  this.selectedBoxWidth  != Math.round( camera3.position.y /0.005 ) ){
                
                this.selectedBoxWidth = Math.round( camera3.position.y / 0.005 );
                camera3.position.y = 6; 
                camera3.position.x = 0;
                camera3.position.z = 0;
            }
            */

            
            $("#panel3").show();
            $("#panel").show();

                    
            
            app.isRedColor = true;
            // recolor_evaluation();
        } else {

        }
    };
    
    
    this.fully_automated_bbox = function(fname) {
        
        if (!settingsControls.FullyAutomatedBbox) {
            return;
        }
        

        var cur_idx = this.fnames.indexOf(fname);
        if(app.cur_frame.annotated){
            return;
        }
         
        
        
        console.log("fully_automated_bbox", fname);
        $("#loading-screen").show();
        
       $("#title-container").text("Generates fully automated prediction...");
        
        $.ajax({
            context: this,
            url: '/fully_automated_bbox',
            data: JSON.stringify({
                fname: fname,
                settingsControls: settingsControls
            }),
            type: 'POST',
            contentType: 'application/json;charset=UTF-8',
            success: function(response) {
                
                if(app.cur_frame.annotated == false){
                    
                    deleteAllBoundingBox(false);
                    var res = response.split("\'").join("\"");

                    res = JSON.parse(res);
                    for (var box_id in res) {
                        if (res.hasOwnProperty(box_id)) {
                            var json_box = res[box_id];
                            var corner1 = new THREE.Vector3(json_box.corner1[1],
                                this.eps,
                                json_box.corner1[0]);
                            var corner2 = new THREE.Vector3(json_box.corner2[1],
                                0,
                                json_box.corner2[0]);
                            
                            
                            var box = createAndDrawBox(corner1,
                                corner2,
                                json_box['angle']);
                            
                            if( isOverlapWithLockedBBOX(box) ){
                            
                                app.cur_frame.last_bbox_id = box.id;
                                
                                scene.remove(box.points);
                                scene.remove(box.boxHelper);
                                delete box;



                            }else{
                            

                                box.is_auto_generated = true;
                                addBox(box);
                            }
                        }
                    }
                    if(box){
                        
                        
                        addObjectRow(box);
                        
                        selectedBox = box;
                    }
                }
                
                
                app.cur_frame.annotated = true;
                
                
                $("#loading-screen").hide();

                $("#title-container").text("");
                
                this.bbox_visualization();
                

                settingsControls.FullyAutomatedBbox = false;

                settingsFolder.updateDisplay();

                
            },
            error: function(error) {
                console.log(error);
                app.cur_frame.annotated = false;
                
                
                $("#loading-screen").hide();

                $("#title-container").text("");
            }
        });


    };
    
    this.delete_other_frames = function(fname){
    
        for(cur_fname in this.frames) {
            
            if(cur_fname != fname){
                // delete this.frames[cur_fname];
            }
        }
    };
    

    this.set_frame = function(fname) {
        //console.log("fname", fname);
        var frame = this.get_frame(fname);
        //this.get_Mask_RCNN_Labels(fname);
        //this.set_controls(fname);
        if (this.cur_frame == frame || this.lock_frame) {
            return;
        }
        if (this.cur_frame) {
            this.write_frame_out();
            this.cur_frame.scene_remove_frame_children();
            this.show_prev_frame = false;
        }

        $("#loading-screen").show();
        $("#container").hide();

        this.bbox_visualization_clearance();
        
        selectedBox = null;
        if (frame) {
            show(frame);
            this.predict_next_frame_bounding_box(this.get_prev_fname(fname));
            $("#loading-screen").hide();
            $("#container").show();

        } else {
            $("#ReloadCurrentFrame").hide();
            $("#title-container").text("Retreiving frame data...");
            $.ajax({
                context: this,
                url: '/getFramePointCloud',
                data: JSON.stringify({
                    fname: fname,
                    settingsControls: settingsControls
                }),
                type: 'POST',
                contentType: 'application/json;charset=UTF-8',
                success: function(response) {
                    var data, res, annotation, bounding_boxes_json, bounding_boxes, box;
                    res = response.split('?');
                    data = res[0].split(',').map(x => parseFloat(x));
                    var frame = new Frame(fname, data);

                    if ( res.length > 1 && res[1].length > 0) {

                        annotation = parsePythonJSON(res[1]);
                        bounding_boxes_json = Object.values(annotation["frame"]["bounding_boxes"]);
                        bounding_boxes = Box.parseJSON(bounding_boxes_json);
                        for (var i = 0; i < bounding_boxes.length; i++) {
                            box = bounding_boxes[i];
                            frame.bounding_boxes.push(box);
                            box.add_text_label();
                            frame.annotated = true;
                        }
                    }
                    
                    $("#title-container").text("");
                    // this.delete_other_frames(fname);
                    this.frames[fname] = frame;

              
                    
                    
                    prev_name = this.get_prev_fname(fname);
                    prev_prev_name = this.get_prev_fname(prev_name);
                    
                    for (var key in this.frames) { // Clear previous frames
                        //if( key != fname && key != prev_name  && key != prev_prev_name ){
                        if( key != fname && key != prev_name ){    
                            deleteFrameByFname(key);
                        }
                    }

                    
                    this.get_Mask_RCNN_Labels(fname);
                    
                    
                    show(frame);


                },
                error: function(error) {
                    
                    $("#title-container").text("");
                    console.log(error);
                }
            });
        }

    };

    this.predict_next_frame_bounding_box = function(fname) {
        if (!enable_bounding_box_tracking) { 
            $("#loading-screen").hide();
            $("#container").show();
            return;
        }
        var cur_idx = this.fnames.indexOf(fname);
        if (cur_idx < 0 ||
            cur_idx >= this.fnames.length - 1 ||
            this.frames[this.fnames[cur_idx + 1]].is_annotated() ||
            !this.frames[this.fnames[cur_idx]] ||
            !this.frames[this.fnames[cur_idx]].is_annotated()) {
            
            console.log("annotated: ", cur_idx,  this.fnames.length);
            
            $("#loading-screen").hide();
            $("#container").show();
            return;
        }
        if (this.fnames[cur_idx].split("/")[0] != this.fnames[cur_idx + 1].split("/")[0]) {
            console.log(this.fnames[cur_idx].split("/")[0], this.fnames[cur_idx+1].split("/")[0]);
            
            $("#loading-screen").hide();
            $("#container").show();
            return;
        }

        //console.log("predict_next_frame_bounding_box", fname);
        
        var next_frame = this.frames[this.fnames[cur_idx + 1]];
        //console.log("predict_next_frame_bounding_box is_annotated", next_frame.is_annotated());

        
        $("#loading-screen").show();
        $("#container").hide(); 
        
        if (!next_frame.annotated) {
            next_frame.annotated = true;
            $("#title-container").text("Generates tracking prediction...");
            $.ajax({
                context: this,
                url: '/predictNextFrameBoundingBoxes',
                data: JSON.stringify({
                    fname: fname,
                    settingsControls: settingsControls
                }),
                type: 'POST',
                contentType: 'application/json;charset=UTF-8',
                success: function(response) {
                    
                    $("#title-container").text("Writing  prediction...");
                    //console.log("response", response);
                    if( response == "error"){
                        
                        //next_frame.annotated = false;
                    }else{
                        var res = response.split("\'").join("\"");
                        res = JSON.parse(res);
                        for (var box_id in res) {
                            if (res.hasOwnProperty(box_id)) {
                            
                                var json_box = res[box_id];
                                var corner1 = new THREE.Vector3(json_box.corner1[1],
                                    this.eps,
                                    json_box.corner1[0]);
                                var corner2 = new THREE.Vector3(json_box.corner2[1],
                                    0,
                                    json_box.corner2[0]);
                                var box = createAndDrawBox(corner1,
                                    corner2,
                                    json_box['angle']);

                                 box.predicted_state =  json_box['predicted_state'];
                                 box.predicted_error =  json_box['predicted_error'];
                                 box.object_id =  json_box['object_id'];
                                 box.id =  box_id;
                                 
                                if( isOverlapWithLockedBBOX(box) ){

                                    app.cur_frame.last_bbox_id = box.id;

                                    scene.remove(box.points);
                                    scene.remove(box.boxHelper);
                                    delete box;



                                }else{


                                    box.is_auto_generated = true;
                                    addBox(box);
                                }
                                
                            }
                        }
                    }

                    
                    gottoObject(1);
                    
                    $("#title-container").text("");
                    this.fully_automated_bbox(fname);
                    
                    
                    $("#loading-screen").hide();
                    $("#container").show();
                    
                    $("#GoToNextFrame").focus();
                    
                    

                },
                error: function(error) {
                    console.log(error);
                    $("#title-container").text("");
                }
            });
        }else{
            
            $("#loading-screen").hide();
            $("#container").show();
            $("#title-container").text("");
            this.fully_automated_bbox(fname);
            gottoObject(1);
        }
    };

    this.get_pointcloud_data = function(fname) {
        if (fname in this.frames) {
            return this.frames[fname].data;
        } else {
            var frame = this.get_frame(fname);
            return frame.data;
        }

    };


    this.getCursor = function() {
        return get3DCoord();
    }

    this.handleBoxRotation = function() {
        if (mouseDown && isRotating) {
            rotatingBox.rotate(this.getCursor());
            rotatingBox.add_timestamp();

            selectedBox = rotatingBox;
            
        }
    }

    this.handleBoxResize = function() {
        if (!isResizing) {
            return;
        }
        if (mouseDown) {
            var cursor = app.getCursor();
            // cursor's y coordinate nudged to make bounding box matrix invertible
            cursor.y -= this.eps;
            resizeBox.resize(cursor);
            resizeBox.add_timestamp();
        } else {
            // evaluator.increment_resize_count();
            predictLabel(resizeBox);
            predictBox = resizeBox;
        }

        selectedBox = resizeBox;


        
        // console.log(resizeBox);
    }

    this.handleBoxMove = function() {
        if (mouseDown && isMoving) {
            selectedBox.translate(this.getCursor());
            selectedBox.changeBoundingBoxColor(selected_color.clone());
            selectedBox.add_timestamp();
            
        }
    }

    this.handleAutoDraw = function() {
        
        if(this.isAutoDrawOn){
            alert("Current annotation is in progress!");
        }
        
        if (autoDrawMode && enable_one_click_annotation && this.isAutoDrawOn == false) {

            this.bbox_visualization_clearance();

            $("#title-container").text("Calculating annotation...");
            $("#loading-screen").show();

            var p = app.getCursor();
            this.isAutoDrawOn = true;
            $.ajax({
                context: this,
                url: '/predictBoundingBox',
                type: 'POST',
                contentType: 'application/json;charset=UTF-8',
                data: JSON.stringify({
                    fname: this.cur_frame.fname,
                    point: app.getCursor(),
                    settingsControls: settingsControls
                }),
                success: function(response) {
                    
                    this.isAutoDrawOn = false;
                    if(response == ""){
                        $("#title-container").text("No annotated point found!");
                 
                    }else{

                        var str = response.replace(/'/g, "\"");
                        var res = JSON.parse(str);
                        var corner1 = new THREE.Vector3(res.corner1[1], this.eps, res.corner1[0]);
                        var corner2 = new THREE.Vector3(res.corner2[1], 0, res.corner2[0]);


                        var newbox = createAndDrawBox(corner1,
                            corner2,
                            res['angle']);

                        if(newbox == null || newbox.boundingBox == false){
                            return false;
                        }

                        if (settingsControls.AutoDeleteExistingBbox) {
                            //console.log("number of bboxes", app.cur_frame.bounding_boxes.length)

                            for (var i_box = app.cur_frame.bounding_boxes.length - 1; i_box >= 0; i_box--) {
                                box = app.cur_frame.bounding_boxes[i_box]
                                //console.log("i_box", i_box);
                                // if( bb_max.x >= p.x && bb_max.z >= p.z && bb_min.x <= p.x && bb_min.z <= p.z ) {

                                if(box.islocked== false){
                                    if (p && containsPoint(box, p) || newbox.boundingBox.intersectsBox(box.boundingBox)) {

                                        // Point is in bounding box

                                        var old_id =  box.id;
                                        newbox.predicted_state = box.predicted_state;
                                        newbox.predicted_error = box.predicted_error;

                                        app.editing_box_id = false;


                                        scene.remove(box.points);
                                        scene.remove(box.boxHelper);

                                        //app.cur_frame.last_bbox_id = box.id;
                                        box.text_label.element.remove();

                                        // deletes corresponding row in object id table
                                        deleteRow(box.id);

                                        // removes selected box from array of currently hovered boxes
                                        hoverBoxes.splice(i_box, 1);

                                        // removes selected box from array of bounding boxes
                                        app.cur_frame.bounding_boxes.splice(i_box, 1);


                                        app.increment_delete_count();
                                        // removes selected box
                                        selectedBox = null;


                                        // Update box id
                                        newbox.id = old_id;
                                    } 
                                }

                            }


                        }

                        addBox(newbox);
                        addObjectRow(newbox);

                        selectedBox = newbox;
                        selectRow(newbox.id);
                        this.bbox_visualization();

                        $("#title-container").text("");


                    }

                    $("#loading-screen").hide();

                    autoDrawModeToggle(false);

                },
                error: function(error) {
                    console.log(error);
                    
                    this.isAutoDrawOn = false;
                    $("#title-container").text("No bounding box is found!");
                    //controls.enabled = true;
                    //controls.update();

                    autoDrawModeToggle(false);
                    $("#loading-screen").hide();

                    //$("#title-container").text("");
                }
            });
        }
    }

    this.get_prev_frame = function() {
        var cur_idx = this.fnames.indexOf(this.cur_frame.fname);
        if (cur_idx == 0 || !(this.fnames[cur_idx - 1] in this.frames)) {
            return null;
        }
        var prev_frame = this.frames[this.fnames[cur_idx - 1]];
        return prev_frame;
    }

    this.show_previous_frame_bounding_box = function() {
        var prev_frame = this.get_prev_frame();
        if (!prev_frame) {
            return;
        }
        console.log("show prev frame: ", this.show_prev_frame);
        if (!this.show_prev_frame) {
            this.show_prev_frame = true;
            prev_frame.scene_add_frame_bounding_box();

        } else if (this.show_prev_frame) {
            this.show_prev_frame = false;
            console.log("remove");
            prev_frame.scene_remove_frame_children();
        }
    }

    this.write_frame_out = function() {
        if (this.cur_frame) {
            this.cur_frame.evaluator.pause_recording();
            var output_frame = this.cur_frame.output();

            
           $("#title-container").text("Saving bounding boxes...");
            
            var output = {
                "frame": output_frame
            };
            var stringifiedOutput = JSON.stringify(output);
            $.ajax({
                url: '/writeOutput',
                data: JSON.stringify({
                    output: {
                        filename: this.cur_frame.fname,
                        settingsControls:settingsControls,
                        file: stringifiedOutput
                    }
                }),
                type: 'POST',
                contentType: 'application/json;charset=UTF-8',
                success: function(response) {
                    console.log("successfully saved output : ");
                    
                   $("#title-container").text("");
                },
                error: function(error) {
                    console.log(error);
                    
            
                   $("#title-container").text("");
                }
            });
        }
    }

    this.render_text_labels = function() {
        if (app.cur_frame) {
            
            //console.log(" app.cur_frame.bounding_boxes",  app.cur_frame.bounding_boxes); 
            for (var i = 0; i < app.cur_frame.bounding_boxes.length; i++) {
                var box = app.cur_frame.bounding_boxes[i];
                
                if (box.text_label) {
                    box.text_label.updatePosition();
                }
            }

            if (app.show_prev_frame) {
                var prev_frame = this.get_prev_frame();
                if (!prev_frame) {
                    return;
                }
                for (var i = 0; i < prev_frame.bounding_boxes.length; i++) {
                    var box = prev_frame.bounding_boxes[i];
                    if (box.text_label) {
                        box.text_label.updatePosition();
                    }
                }
            }
        }
    }

    this.generate_new_box_id = function() {
        if (app.cur_frame) {
            
            var box_ids = [];
            for (var i = 0; i < app.cur_frame.bounding_boxes.length; i++) {
                app.cur_frame.bounding_boxes[i].id  = parseInt(app.cur_frame.bounding_boxes[i].id )
                box_ids.push( app.cur_frame.bounding_boxes[i].id );
            }
            //console.log(box_ids);
            
            if(box_ids.includes(app.cur_frame.last_bbox_id ) == false){
                return app.cur_frame.last_bbox_id;
            }else if(box_ids.includes( app.cur_frame.bounding_boxes.length ) == false){
                return app.cur_frame.bounding_boxes.length;
            }else if (box_ids.length > 0) {
                return Math.max.apply(Math, box_ids) + 1;
            }
        }
        return 99;
    }

    this.get_Mask_RCNN_Labels = function(fname) {
        if (!enable_mask_rcnn || this.frames[fname].mask_rcnn_indices.length > 0) {
            
            
            return;
        }
        this.lock_frame = true;
        
        $("#title-container").text("Retreiving point annotation...");
        $.ajax({
            context: this,
            url: '/getMaskRCNNLabels',
            data: JSON.stringify({
                fname: fname,
                settingsControls: settingsControls
            }),
            type: 'POST',
            contentType: 'application/json;charset=UTF-8',
            success: function(response) {
                var l = response.length - 1;
                maskRCNNIndices = response.substring(1, l).split(',').map(Number);
                // console.log(maskRCNNIndices);
                // console.log(response);
                this.frames[fname].mask_rcnn_indices = maskRCNNIndices;
                highlightPoints(maskRCNNIndices);
                updateMaskRCNNImagePanel();
                this.lock_frame = false;
                
                $("#title-container").text("");
                
                this.predict_next_frame_bounding_box(this.get_prev_fname(fname));
                

            },
            error: function(error) {
                console.log(error);
                
                
                $("#title-container").text("Failed to load point-wise label!");
                $("#loading-screen").hide();
                $("#container").show();
                
                this.lock_frame = false;
                delete this.frames[fname];
                //this.set_frame(fname);
            }
        });
    }

    this.pause_3D_time = function() {
        if (this.cur_frame && isRecording) {
            this.cur_frame.evaluator.pause_3D_time();
        }
    }
    this.increment_label_count = function() {
        if (this.cur_frame && isRecording) {
            this.cur_frame.evaluator.increment_label_count();
        }
    }

    this.decrement_label_count = function() {
        if (this.cur_frame && isRecording) {
            this.cur_frame.evaluator.decrement_label_count();
        }
    }

    this.increment_add_box_count = function() {
        if (this.cur_frame && isRecording) {
            this.cur_frame.evaluator.increment_add_box_count();
        }
    }

    this.increment_translate_count = function() {
        if (this.cur_frame && isRecording) {
            this.cur_frame.evaluator.increment_translate_count();
        }
    }

    this.increment_rotate_count = function() {
        if (this.cur_frame && isRecording) {
            this.cur_frame.evaluator.increment_rotate_count();
        }
    }

    this.increment_rotate_camera_count = function() {
        if (this.cur_frame && isRecording) {
            this.cur_frame.evaluator.increment_rotate_camera_count(camera.rotation.z);
        }
    }

    this.increment_resize_count = function() {
        if (this.cur_frame && isRecording) {
            this.cur_frame.evaluator.increment_resize_count(camera.rotation.z);
        }
    }

    this.increment_delete_count = function() {
        if (this.cur_frame && isRecording) {
            this.cur_frame.evaluator.increment_delete_count();
        }
    }

    this.resume_3D_time = function() {
        if (this.cur_frame && isRecording) {
            this.cur_frame.evaluator.resume_3D_time();
        }
    }

    this.pause_recording = function() {
        if (this.cur_frame && isRecording) {
            this.cur_frame.evaluator.pause_recording();
        }
    }

    this.resume_recording = function() {
        if (this.cur_frame && isRecording) {
            this.cur_frame.evaluator.resume_recording();
        }
    }

    this.set_controls = function(fname) {
        var i = this.controls[fname];
        console.log("asdf, ", i);
        if (i == 0) {
            enable_predict_label = false;
            enable_mask_rcnn = false;
            enable_one_click_annotation = false;
            enable_bounding_box_tracking = false;
        } else if (i == 1) {
            enable_predict_label = true;
            enable_mask_rcnn = true;
            enable_one_click_annotation = false;
            enable_bounding_box_tracking = false;
        } else if (i == 2) {
            enable_predict_label = false;
            enable_mask_rcnn = false;
            enable_one_click_annotation = true;
            enable_bounding_box_tracking = false;
        } else if (i == 3) {
            enable_predict_label = false;
            enable_mask_rcnn = false;
            enable_one_click_annotation = false;
            enable_bounding_box_tracking = true;
        } else if (i == 4) {
            enable_predict_label = true;
            enable_mask_rcnn = true;
            enable_one_click_annotation = true;
            enable_bounding_box_tracking = true;
        } else if (i == 5) {
            enable_predict_label = true;
            enable_mask_rcnn = true;
            enable_one_click_annotation = true;
            enable_bounding_box_tracking = true;
        }
    }

}

function parsePythonJSON(json) {
    return JSON.parse(json.split("\'").join("\"").split("False").join("false").split("True").join("true"));
}

function show(frame) {
    var initPointCloud;

    
    var is_object_table_visible = $("#object-table").is(":visible");

    
    if (app.cur_frame) {
        clearObjectTable();
    }
    app.cur_frame = frame;
    if (app.cur_pointcloud == null) {
        initPointCloud = true;
    }


    for (var i = scene.children.length - 1; i >= 0; i--) {
        //scene.remove(scene.children[i]);
    }

    // add pointcloud to scene
    generatePointCloud();

    if (initPointCloud) {
        scene.add(app.cur_pointcloud);



        var dotGeometry = new THREE.Geometry();
        dotGeometry.vertices.push(new THREE.Vector3(0, 0, 0));
        var dotMaterial = new THREE.PointsMaterial({
            size: 5,
            sizeAttenuation: false
        });
        var dot = new THREE.Points(dotGeometry, dotMaterial);
        scene.add(dot);


        animate();
    }
    app.cur_frame.scene_add_frame_children();
    loadObjectTable();
    
     var _current = app.move2D;
    switchMoveMode();

    switch2DMode();
    app.move2D = _current;
    updateCountBBOX();
    
    $("#ReloadCurrentFrame").show();

    if(app.move2D == true){
        
        switch2DMode();
    }
    
    if(is_object_table_visible){

        switch2DMode();
        
        
        eventFire(document.getElementById('objectIDs'), 'click');
        
        $("#GoToNextFrame").focus();
        
        
    }
    
    //camera3.rotateZ(3.14);



    // if (isRecording) {
    // 	toggleRecord(event);
    // }
}