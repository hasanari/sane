function Box(anchor, cursor, angle, boundingBox, boxHelper) {
    this.id = app.generate_new_box_id(); // id (int) of Box
    this.object_id = 'car'; // object id (string)
    this.color = hover_color.clone(); // color of corner points
    this.angle = angle; // orientation of bounding box
    this.anchor = anchor; // point where bounding box was created
    this.cursor = cursor.clone(); // cursor
    this.islocked = false;
    this.is_auto_generated =false;
    this.initialcursor=false;
    this.height = 0; // cursor
    
    this.predicted_state = [0, 0, 1, 1, 1, 1]; // x, y, , vx, vy, ax, ay
    this.predicted_error =  [0, 0, 0, 0, 0, 0]; // x, y, , vx, vy, ax, ay
    
    this.added = false; // (boolean) whether the box has been added to boundingboxes
    this.boundingBox = boundingBox; // Box3; sets the size of the box
    this.boxHelper = boxHelper; // BoxHelper; helps visualize the box
    this.geometry = new THREE.Geometry(); // geometry for corner/rotating points

    this.settingsControls = $.extend( {}, settingsControls);
    
    // visualizes the corners (in the non-rotated coordinates) of the box
    
    this.pointMaterial = new THREE.PointsMaterial({
        size: 8,
        sizeAttenuation: false,
        vertexColors: THREE.VertexColors
    });
    this.points = new THREE.Points( this.geometry, this.pointMaterial );
    this.points.frustumCulled = false; // allows 
    this.timestamps = [];
    
    this.colors = []; // colors of the corner points

    // add colors to points geometry
    for (var i = 0; i < 6; i++) {
        this.colors.push( this.color.clone().multiplyScalar( 7 ) );
    }
    this.geometry.colors = this.colors;
    
    // this.middlePointMaterial = new THREE.PointsMaterial( { size: 3.0, sizeAttenuation: false, vertexColors: THREE.VertexColors } );
    
    // order of corners is max, min, topleft, bottomright
    this.geometry.vertices.push(anchor);
    this.geometry.vertices.push(cursor);
    this.geometry.vertices.push(anchor.clone());
    this.geometry.vertices.push(cursor.clone());
    this.geometry.vertices.push(getCenter(anchor.clone(), cursor.clone()));
    
    
    this.hasPredictedLabel = false;
    this.text_label;

    this.get_center = function() {
        var center3D = getCenter(this.geometry.vertices[0], this.geometry.vertices[1]);
        return new THREE.Vector2(center3D.z, center3D.x);
    }
   
    // method for resizing bounding box given cursor coordinates
    // 
    // since BoxHelper3 draws a box in the same orientation as that of the point cloud, 
    // we take the anchor and cursor, rotate them by the angle of the camera, draw the box, 
    // then rotate the box back
    
    this.changesOnLatestResize = function(cursor){
    
        var v1 = cursor.clone();
        var vInit = this.initialcursor;

        if(vInit){
            return {x:( v1.x - vInit.x ), y:(v1.y - vInit.y), z:(v1.z - vInit.z)};
        }
        return {x:0.0, y:0.0, z:0.0 };
    }
    
    this.resize = function(cursor) {
        // checks and executes only if anchor does not overlap with cursor to avoid 0 determinant
        if (cursor.x != this.anchor.x && cursor.y != this.anchor.y && cursor.z != this.anchor.z) {

            var v1 = cursor.clone();
            var vInit = this.initialcursor;
            var v2 = this.anchor.clone();
            
   
            v1.y = 0;
            v2.y = 0;
            
            // rotate cursor and anchor
            rotate(v1, v2, this.angle);

            // calculating corner points and rotating point
            var minVector = getMin(v1, v2);
            var maxVector = getMax(v1, v2);
            var topLeft = getTopLeft(v1, v2);
            var bottomRight = getBottomRight(v1, v2);
            var topCenter = getCenter(topLeft, maxVector);
            var bottomCenter = getCenter(minVector, bottomRight);

            // need to do this to make matrix invertible
            maxVector.y = 0.00001; 

            // setting bounding box limits
            this.boundingBox.set(minVector.clone(), maxVector.clone());

            // rotate BoxHelper back
            this.boxHelper.rotation.y = this.angle;

            // setting y coordinate back to zero since we are done with drawing
            maxVector.y = 0;

            // rotate back the corner points
            rotate(minVector, maxVector, -this.angle);
            rotate(topLeft, bottomRight, -this.angle);
            rotate(topCenter, bottomCenter, -this.angle);

            // set updated corner points used to resize box
            this.geometry.vertices[0] = maxVector.clone();
            this.geometry.vertices[1] = minVector.clone();
            this.geometry.vertices[2] = topLeft.clone();
            this.geometry.vertices[3] = bottomRight.clone();
            this.geometry.vertices[4] = bottomCenter.clone();

            // tell scene to update corner points
            this.geometry.verticesNeedUpdate = true;
        }
    }

    // method to rotate bounding box by clicking and dragging rotate point, 
    // which is the top center point on the bounding box
    this.rotate = function(cursor) {
        // get corner points
        var maxVector = this.geometry.vertices[0].clone();
        var minVector = this.geometry.vertices[1].clone();
        var topLeft = this.geometry.vertices[2].clone();
        var bottomRight = this.geometry.vertices[3].clone();
        var topCenter = getCenter(maxVector, topLeft);
        var bottomCenter = this.geometry.vertices[4].clone();

        // get relative angle of cursor with respect to 
        var center = getCenter(maxVector, minVector);
        var angle = getAngle(center, bottomCenter, cursor, topCenter);

        this.innerRotate(angle);
        
    }
    this.innerRotate = function(angle) {
        // get corner points
        var maxVector = this.geometry.vertices[0].clone();
        var minVector = this.geometry.vertices[1].clone();
        var topLeft = this.geometry.vertices[2].clone();
        var bottomRight = this.geometry.vertices[3].clone();
        var topCenter = getCenter(maxVector, topLeft);
        var bottomCenter = this.geometry.vertices[4].clone();

        // get relative angle of cursor with respect to 
        var center = getCenter(maxVector, minVector);
        // var angle = getAngle(center, bottomCenter, cursor, topCenter);

        // update angle of Box and bounding box
        this.angle = this.angle + angle;
        this.boxHelper.rotation.y = this.angle;

        // rotate and update corner points
        rotate(minVector, maxVector, -angle);
        rotate(topLeft, bottomRight, -angle);
        rotate(topCenter, bottomCenter, -angle);

        this.geometry.vertices[0] = maxVector.clone();
        this.geometry.vertices[1] = minVector.clone();
        this.geometry.vertices[2] = topLeft.clone();
        this.geometry.vertices[3] = bottomRight.clone();
        this.geometry.vertices[4] = bottomCenter.clone();

        // tell scene to update corner points
        this.geometry.verticesNeedUpdate = true;
        
    }


    // method to translate bounding box given a reference point
    this.translate = function(v) {
        // get difference in x and z coordinates between cursor when 
        // box was selected and current cursor position
        var dx = v.x - this.cursor.x;
        var dz = v.z - this.cursor.z;

        // update all points related to box by dx and dz
        this.anchor.x += dx;
        this.anchor.z += dz;
        this.cursor = v.clone();
        for (var i = 0; i < this.geometry.vertices.length; i++) {
            var p = this.geometry.vertices[i];
            p.x += dx;
            p.z += dz;
        }

        // shift bounding box given new corner points
        var maxVector = this.geometry.vertices[0].clone();
        var minVector = this.geometry.vertices[1].clone();
        var topLeft = this.geometry.vertices[2].clone();
        var bottomRight = this.geometry.vertices[3].clone();
        var topCenter = getCenter(maxVector, topLeft);
        var bottomCenter = this.geometry.vertices[4].clone();

        rotate(maxVector, minVector, this.angle);
        rotate(topLeft, bottomRight, this.angle);
        rotate(topCenter, bottomCenter, this.angle);

        // need to do this to make matrix invertible
        maxVector.y += 0.0000001; 

        this.boundingBox.set(minVector, maxVector);

        // tell scene to update corner points
        this.geometry.verticesNeedUpdate = true;
    }

    // method to highlight box given cursor
    this.select = function(cursor) {
        selectedBox = this;
        if (this && cursor) {
            selectedBox.cursor = cursor;
        }
        updateHoverBoxes(cursor);
        // this.changeBoundingBoxColor(new THREE.Color( 0,0,7 ) );
        this.changeBoundingBoxColor(selected_color);
    }


    // changes and updates a box's point's color given point index and color
    this.changePointColor = function(idx, color) {
        this.colors[idx] = color;
        this.geometry.colorsNeedUpdate = true;
    }
    // method to change color of bounding box
    this.changeBoundingBoxColor = function(color) {
        boxHelper.material.color.set(color);
    }

    this.output = function() {
        return new OutputBox(this);
    }

    this.get_cursor_distance_threshold = function() {
        return Math.min(distance2D(this.geometry.vertices[0], this.geometry.vertices[2]),
            distance2D(this.geometry.vertices[0], this.geometry.vertices[1])) / 4;
    }

    this.set_box_id = function(box_id) {
        if (typeof(box_id) == 'string') {
            box_id = parseInt(box_id);
        }
        this.id = box_id;
        this.text_label.setHTML(this.id.toString());
    }

    this.add_timestamp = function() {
        this.timestamps.push(Date.now());
    }

    this.update_text_label = function(old_id) {
        $('#text-label'+old_id).text(this.id.toString());
        $('#text-label'+old_id).attr("id",'text-label'+this.id.toString());
        

    }
    
    this.add_text_label = function() {
        var text = this.create_text_label();
        text.setHTML(this.id.toString());
        text.setParent(this.boxHelper);
        container.appendChild(text.element);
        this.text_label = text;
    }

    this.create_text_label = function() {
        var div = document.createElement('div');
        div.id = 'text-label'+this.id.toString();
        div.className = 'text-label';
        div.style.position = 'absolute';
        div.style.width = 100;
        div.style.height = 100;
        div.innerHTML = "hi there!";
        div.style.top = -1000;
        div.style.left = -1000;
    
        var _this = this;
    
        return {
          element: div,
          parent: false,
          position: new THREE.Vector3(0,0,0),
          setHTML: function(html) {
            this.element.innerHTML = html;
          },
          setParent: function(threejsobj) {
            this.parent = threejsobj;
          },
          updatePosition: function() {
            if (this.parent) {
              this.position.copy(this.parent.position);
            }            
            var coords2d = this.get2DCoords(this.position, camera);
            this.element.style.left = coords2d.x + 'px';
            this.element.style.top = coords2d.y + 'px';
          },
          get2DCoords: function(position, camera) {
            var vector = position.project(camera);
            vector.x = (vector.x + 1)/2 * window.innerWidth;
            vector.y = -(vector.y - 1)/2 * window.innerHeight;
            return vector;
          }
        };
    }
}



// gets angle between v1 and v2 with respect to origin
//
// v3 is an optional reference point that should be v1's reflection about the origin, 
// but is needed to get the correct sign of the angle
function getAngle(origin, v1, v2, v3) {
    v1 = v1.clone();
    v2 = v2.clone();
    origin = origin.clone();
    v1.sub(origin);
    v2.sub(origin);
    v1.y = 0;
    v2.y = 0;
    v1.normalize();
    v2.normalize();

    var angle = Math.acos(Math.min(1.0, v1.dot(v2)));
    if (v3) {
        v3 = v3.clone();
        v3.sub(origin);

        // calculates distance between v1 and v2 when v1 is rotated by angle
        var temp1 = v1.clone();
        rotate(temp1, v3.clone(), angle);
        var d1 = distance2D(temp1, v2);

        // calculates distance between v1 and v2 when v1 is rotated by -angle
        var temp2 = v1.clone();
        rotate(temp2, v3.clone(), -angle);
        var d2 = distance2D(temp2, v2);
        


        // compares distances to determine sign of angle
        if (d2 > d1) {
            angle = -angle;
        }
    }

    return angle;
}


// highlights closest corner point that intersects with cursor
function highlightCorners() {
    // get closest intersection with cursor
    var intersection = intersectWithCorner();
    if (intersection) {
        // get closest point and its respective box
        var box = intersection[0];
        var p = intersection[1];

        // get index of closest point
        var closestIdx = closestPoint(p, box.geometry.vertices);

        // if there was a previously hovered box, change its color back to red
        if (hoverBox) {
            // hoverBox.changePointColor(hoverIdx, new THREE.Color(7, 0, 0));
            hoverBox.changePointColor(hoverIdx, hover_color.clone());
        }

        // update hover box
        hoverBox = box;
        hoverIdx = closestIdx;
        // hoverBox.changePointColor(hoverIdx, new THREE.Color(0, 0, 7));
        hoverBox.changePointColor(hoverIdx, selected_color.clone());

    } else {

        // change color of previously hovered box back to red
        if (hoverBox) {
            // hoverBox.changePointColor(hoverIdx, new THREE.Color(7, 0, 0));
            hoverBox.changePointColor(hoverIdx, hover_color.clone());
        }

        // set hover box to null since there is no intersection
        hoverBox = null;
    }
}




// highlights closest corner point that intersects with cursor
function highlightCornersTopView() {
    // get closest intersection with cursor
    if(!app.selectedBox){
        return
    }
    
    var intersection = intersectionWithCornerTopView(app.selectedBox);
    if (intersection) {
        // get closest point and its respective box
        var box = intersection[0];
        var p = intersection[1];
        

        // get index of closest point
        var closestIdx = closestPoint(p, box.geometry.vertices);

        
        // if there was a previously hovered box, change its color back to red
        if (hoverBox) {
            // hoverBox.changePointColor(hoverIdx, new THREE.Color(7, 0, 0));
            hoverBox.changePointColor(hoverIdx, hover_color.clone());
        }

        // update hover box
        hoverBox = box;
        hoverIdx = closestIdx;
        // hoverBox.changePointColor(hoverIdx, new THREE.Color(0, 0, 7));
        hoverBox.changePointColor(hoverIdx, selected_color.clone());

    } else {

        // change color of previously hovered box back to red
        if (hoverBox) {
            // hoverBox.changePointColor(hoverIdx, new THREE.Color(7, 0, 0));
            hoverBox.changePointColor(hoverIdx, hover_color.clone());
        }

        // set hover box to null since there is no intersection
        hoverBox = null;
    }
    
    
    
    
        
        
}
    
function changeCursor(is_Moving, cursor, box){


    if(is_Moving == false){
        $("body").css("cursor", "default");
    }else if (cursor && containsPoint(box, cursor)) {
        
        $("body").css("cursor", "move");
    }else{
        $("body").css("cursor", "default");
    
    }

}




// method to add box to boundingBoxes and object id table
// should only be called when you physically draw a box, 
// not for loading a frame
function addBox(box) {

    app.cur_frame.bounding_boxes.push(box);
    addObjectRow(box);
    box.add_text_label();
    
    
    updateCountBBOX();
    
}

function stringifyBoundingBoxes(boundingBoxes) {
    var outputBoxes = [];
    for (var i = 0; i < boundingBoxes.length; i++) {
        outputBoxes.push(new OutputBox(boundingBoxes[i]));
    }
    return outputBoxes;
}

function createBox(anchor, v, angle) {
    
    newBox = null;
    if( distance2D(v, anchor) > 0.2){
        newBoundingBox = new THREE.Box3(v, anchor);
        newBoxHelper = new THREE.Box3Helper( newBoundingBox, 0xffff00 );
        newBox = new Box(anchor, v, angle, newBoundingBox, newBoxHelper);
        newBox.resize(v);
    }
    

    return newBox;
}

function createAndDrawBox(anchor, v, angle) {
    var newBox = createBox(anchor, v, angle);
    
    drawBox(newBox);
    return newBox;
}

function drawBox(box) {
    
    if(box != null){
    
        scene.add(box.points);
        scene.add(box.boxHelper);


        updateAllObjectIds();
    }
    
}

// deletes selected box when delete key pressed
function deleteSelectedBox() {
    
            
    if (app.editing_box_id) {return;}
    var boundingBoxes = app.cur_frame.bounding_boxes;
    if (selectedBox && selectedBox.islocked==false && mouseEvent == "up") {
        
        app.bbox_visualization_clearance();
        
        delete_one_box(selectedBox);
        
        updateAllObjectIds();
    }
    
    updateCountBBOX();
}

function isOverlapWithLockedBBOX(check_box){

    
    var lockedBox = [];
    for (var i_box = app.cur_frame.bounding_boxes.length - 1; i_box >= 0; i_box--) {
        box = app.cur_frame.bounding_boxes[i_box]
        if(box.islocked){
            lockedBox.push(box);
        }
    }
    
    for (var i_box = lockedBox.length - 1; i_box >= 0; i_box--) {
     
        box = lockedBox[i_box]
        if( check_box.boundingBox.intersectsBox(box.boundingBox) || box.boundingBox.intersectsBox(check_box.boundingBox) ){ 
            // within the intersection
            return true;
        }
        
    }
    
    return false;

}

function deleteAllBoundingBox(delete_auto_generated_only){


    app.bbox_visualization_clearance();
    for (var i_box = app.cur_frame.bounding_boxes.length - 1; i_box >= 0; i_box--) {
        box = app.cur_frame.bounding_boxes[i_box]
        
        if(box.islocked== false){
        
            if(delete_auto_generated_only==false || box.is_auto_generated){

               delete_one_box(box);
            }
        }

    }
    
    updateAllObjectIds();
    
    updateCountBBOX();
    
}

function delete_one_box(box){
    // Point is in bounding box
    app.editing_box_id = false;


    scene.remove(box.points);
    scene.remove(box.boxHelper);
    box.text_label.element.remove();

    // deletes corresponding row in object id table
    app.cur_frame.last_bbox_id = box.id;
    deleteRow(box.id);
    
    
    // removes selected box from array of currently hovered boxes
    for (var i = 0; i < hoverBoxes.length; i++) {
        if (hoverBoxes[i] == box) {
            hoverBoxes.splice(i, 1);
            break;
        }
    }

    // removes selected box from array of bounding boxes
    for (var i = 0; i <  app.cur_frame.bounding_boxes.length; i++) {
        if ( app.cur_frame.bounding_boxes[i] == box) {
             app.cur_frame.bounding_boxes.splice(i, 1);
            break;
        }
    }


    app.increment_delete_count();
    // removes selected box
    selectedBox = null;

}


Box.parseJSON = function(json_boxes) {
    var bounding_boxes = [], box;
    var json_box, center, top_right, bottom_left;
    var w, l, cx, cy, angle;
    if (!Array.isArray(json_boxes)) {
        json_boxes = [json_boxes];
    }
    for (var i = 0; i < json_boxes.length; i++) {
        json_box = json_boxes[i];
        w = json_box['width'];
        l = json_box['length'];
        cx = json_box['center']['x'];
        cy = json_box['center']['y'];
        angle = json_box['angle'];
        height = json_box['height'];
        is_auto_generated = json_box['is_auto_generated'];
        islocked = json_box['islocked'];
        

        center = new THREE.Vector3(cy, 0, cx);
        top_right = new THREE.Vector3(cy + l / 2, app.eps, cx + w / 2);
        bottom_left = new THREE.Vector3(cy - l / 2, 0, cx - w / 2);
        
        
        
        //points  = json_box['points']; 
        //console.log("points", points);
        //top_right = new THREE.Vector3(points[0].x, points[0].y+app.eps, points[0].z);
        //bottom_left = new THREE.Vector3(points[1].x, points[1].y, points[1].z);
        
        // rotate cursor and anchor
        rotate(top_right, bottom_left, -angle);
        box = createBox(top_right, bottom_left, angle);
        
        if( box != null ){
        
            if (json_box.hasOwnProperty('box_id')) {
                box.id = json_box.box_id;
            }
            box.is_auto_generated = is_auto_generated;
            
            box.object_id =  json_box['object_id'];
            box.predicted_state =  json_box['predicted_state'];
            box.predicted_error =  json_box['predicted_error'];
            box.height =  json_box['height'];
            box.islocked = json_box['islocked'];
            box.timestamps = json_box["timestamps"];
            bounding_boxes.push(box);
            
            if(box.islocked){
                box.pointMaterial.size =0;
            }
                
        }
        //console.log("output: ", box, json_box);
    }
    return bounding_boxes;
}


function OutputBox(box) {
    /*
    
            this.geometry.vertices[0] = maxVector.clone();
            this.geometry.vertices[1] = minVector.clone();
            this.geometry.vertices[2] = topLeft.clone();
            this.geometry.vertices[3] = bottomRight.clone();
            this.geometry.vertices[4] = bottomCenter.clone();
            
    */
    var v1 = box.geometry.vertices[0];
    var v2 = box.geometry.vertices[1];
    var v3 = box.geometry.vertices[2];
    var center = getCenter(v1, v2);
    
    this.points = [v1.clone(), v2.clone(), v3.clone()];
    this.center = new THREE.Vector2(center.z, center.x);
    this.width = distance2D(v2, v3);
    this.length = distance2D(v1, v3);
    
    
    this.box_id = box.id;
    this.islocked = box.islocked;
    this.is_auto_generated = box.is_auto_generated;
    this.height = box.height;
    this.angle = box.angle;
    this.object_id = box.object_id;
    this.settingsControls = box.settingsControls;
    this.timestamps = box.timestamps;
    
    this.predicted_error = box.predicted_error;
    
    var v_x = box.predicted_state[0]-center.z;
    var v_y = box.predicted_state[1]-center.x;
    this.predicted_state = [box.predicted_state[0], box.predicted_state[1], v_x, v_y, box.predicted_state[2] - v_x, box.predicted_state[3] - v_y ];
}