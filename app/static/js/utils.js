function create_box_from_annot(opt_box, surrounding_eps){

    // Making enlarge boxes
    w = opt_box['width'] + surrounding_eps ;
    l = opt_box['length'] +surrounding_eps ;
    cx = opt_box['center'].x;
    cy = opt_box['center'].y;
    angle = opt_box['angle'];

    center = new THREE.Vector3(cy, 0, cx);
    top_right = new THREE.Vector3(cy + l / 2, app.eps, cx + w / 2);
    bottom_left = new THREE.Vector3(cy - l / 2, 0, cx - w / 2);

    // rotate cursor and anchor
    rotate(top_right, bottom_left, -angle);
    return createBox(top_right, bottom_left, angle);

}

function update_point_size(){

    for(var i=0; i < app.cur_frame.bounding_boxes.length; i++){
        var box = app.cur_frame.bounding_boxes[i];
        
        if(box.islocked){
        
            box.pointMaterial.size=0;
        }else{
            box.pointMaterial.size=8;
        }
        
    }
}

function get_point_inside_box(vertices, opt_box, mask_indices, surrounding_eps){


    bounding_boxes_selected = create_box_from_annot(opt_box, surrounding_eps);

    
    sidebar_STRIDE = 4
    pointsIn = []; 
    py =[];
    
    masked_indices =[];
    center = bounding_boxes_selected.get_center();
    
    
    var point_selected = 0;
    var k = 0;
    
    /*
    var pointFiltered = indexedPoints.nearest({x:center.x, z:center.y}, 10000, 5);
    
    console.log(center);
    console.log(pointFiltered);
    
    for(var i=0; i< pointFiltered.length; i++ ){
        
        var v = new THREE.Vector3( pointFiltered[i][0].z, 0, pointFiltered[i][0].x );
            
        if (v && containsPoint(bounding_boxes_selected, v)) {
            
            
            //console.log(pointFiltered[i][0], center,  pointFiltered[i][0].z - center.x ,  pointFiltered[i][0].x - center.z)
            
            
            pointsIn.push( pointFiltered[i][0].x - center.z ); 
            pointsIn.push( pointFiltered[i][0].z - center.x );  
            pointsIn.push( pointFiltered[i][0].y ); 
            pointsIn.push( 0 ); 
            
            py.push(pointFiltered[i][0].y);
            
            if( mask_indices.includes( pointFiltered[i][0].idx ) ){
                masked_indices.push(point_selected);
            }
            
            point_selected++;
        }
        
        k++; 
    }
    console.log("pointsIn", pointsIn);
    
    return [pointsIn, py, masked_indices];
    */
    for ( var i = 0, l = vertices.length / sidebar_STRIDE; i < l; i ++ ) {
        // creates new vector from a cluster and adds to geometry
        var v = new THREE.Vector3( vertices[ sidebar_STRIDE * k + 1 ], 
            0, vertices[ sidebar_STRIDE * k ] );
        
        if (v && containsPoint(bounding_boxes_selected, v)) {
            pointsIn.push(vertices[ sidebar_STRIDE * k + 0 ] - center.z ); 
            pointsIn.push(vertices[ sidebar_STRIDE * k + 1 ] - center.x); 
            pointsIn.push( vertices[ sidebar_STRIDE * k + 2 ] ); 
            pointsIn.push( vertices[ sidebar_STRIDE * k + 3 ] ); 
            
            py.push( vertices[ sidebar_STRIDE * k + 2 ] );
            
            if( mask_indices.includes(k) ){
                masked_indices.push(point_selected);
            }
            
            point_selected++;
        }
        
        k++; 
    }
    
    
    return [pointsIn, py, masked_indices];
}

function calculateMean(arr) {
    var total = 0;
    for (var i = 0; i< arr.length; i++) {
        total += arr[i];
    }
    return total / arr.length;
}

/* https://stackoverflow.com/questions/3972014/get-caret-position-in-contenteditable-div */
function getCaretPosition(editableDiv) {
  var caretPos = 0,
    sel, range;
  if (window.getSelection) {
    sel = window.getSelection();
    if (sel.rangeCount) {
      range = sel.getRangeAt(0);
      if (range.commonAncestorContainer.parentNode == editableDiv) {
        caretPos = range.endOffset;
      }
    }
  } else if (document.selection && document.selection.createRange) {
    range = document.selection.createRange();
    if (range.parentElement() == editableDiv) {
      var tempEl = document.createElement("span");
      editableDiv.insertBefore(tempEl, editableDiv.firstChild);
      var tempRange = range.duplicate();
      tempRange.moveToElementText(tempEl);
      tempRange.setEndPoint("EndToEnd", range);
      caretPos = tempRange.text.length;
    }
  }
  return caretPos;
}


function standardDeviation(arr) {
    var mean = calculateMean(arr);
    var variance = 0;
    for (var i = 0; i < arr.length; i++) {
        variance += Math.pow(arr[i] - mean, 2);
    }
    variance = variance / arr.length;
    return Math.pow(variance, 0.5);
}

function filter(arr, mean, thresh) {
    var result = [];
    for (var i = 0; i< arr.length; i++) {
        if (Math.abs(arr[i] - mean) < thresh) {
            result.push(arr[i]);
        }
    }
    return result;
}

function getMinElement(arr) {
    var min = Number.POSITIVE_INFINITY;
    for (var i = 0; i< arr.length; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    return min;
}

function isInsideTheVisBox(event){

    var panelMouse = document.getElementById("panel").getBoundingClientRect(); 
    if( event.clientX >= panelMouse.x && event.clientX <= ( panelMouse.x+ panelMouse.width) ){
       
        if( event.clientY >= panelMouse.y && event.clientY <= ( panelMouse.y+ panelMouse.height) ){
            return true;
        }
    }
    
    return false;
    
}

function get3DCoord() {
    mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
    mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
    mouse.z = 0.5;
    mouse.unproject( camera );

    var dir = mouse.sub( camera.position ).normalize();
    var distance = - camera.position.y / dir.y;
    var pos = camera.position.clone().add( dir.multiplyScalar( distance ) );
    return pos;
}


function get3DCoordTopView() {
    
    
    var panelMouse = document.getElementById("panel").getBoundingClientRect();   
    mouseTopView.x = ( (event.clientX-panelMouse.x) / panelMouse.width ) * 2 - 1;
    mouseTopView.y = - ( (event.clientY-panelMouse.y) / panelMouse.height ) * 2 + 1;
    mouseTopView.z = 0.5;
    mouseTopView.unproject( camera3 );

    var dir = mouseTopView.sub( camera3.position ).normalize();
    var distance = - camera3.position.y / dir.y;
    var pos = camera3.position.clone().add( dir.multiplyScalar( distance ) );
    return pos;
}


function getMaxElement(arr) {
    var max = Number.NEGATIVE_INFINITY;
    for (var i = 0; i< arr.length; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

function getMin(v1, v2) {
    return new THREE.Vector3(Math.min(v1.x, v2.x), 
                             Math.min(v1.y, v2.y), 
                             Math.min(v1.z, v2.z))
}

function getMax(v1, v2) {
    return new THREE.Vector3(Math.max(v1.x, v2.x), 
                             Math.max(v1.y, v2.y), 
                             Math.max(v1.z, v2.z))
}

function getTopLeft(v1, v2) {
    return new THREE.Vector3(Math.min(v1.x, v2.x), 
                             Math.max(v1.y, v2.y), 
                             Math.max(v1.z, v2.z))
}

function getBottomRight(v1, v2) {
    return new THREE.Vector3(Math.max(v1.x, v2.x), 
                             Math.min(v1.y, v2.y), 
                             Math.min(v1.z, v2.z))
}

function getCenter(v1, v2) {
    return new THREE.Vector3((v1.x + v2.x) / 2.0, 0.0, (v1.z + v2.z) / 2.0);
}

function rotate(v1, v2, angle) {
    center = getCenter(v1, v2);
    v1.sub(center);
    v2.sub(center);
    var temp1 = v1.clone();
    var temp2 = v2.clone();
    v1.x = Math.cos(angle) * temp1.x - Math.sin(angle) * temp1.z;
    v2.x = Math.cos(angle) * temp2.x - Math.sin(angle) * temp2.z;

    v1.z = Math.sin(angle) * temp1.x + Math.cos(angle) * temp1.z;
    v2.z = Math.sin(angle) * temp2.x + Math.cos(angle) * temp2.z;

    v1.add(center);
    v2.add(center);
}

function getOppositeCorner(idx) {
    if (idx == 0) {return 1;}
    if (idx == 1) {return 0;}
    if (idx == 2) {return 3;}
    return 2;
}


function containsPoint(box, v) {
    var center = getCenter(box.boundingBox.max, box.boundingBox.min);
    var diff = v.clone();
    diff.sub(center);
    var v1 = v.clone();
    var v2 = center;
    v2.sub(diff);
    rotate(v1, v2, box.angle);
    return box.boundingBox.containsPoint(v2);
}


function intersectWithCorner() {
    var boundingBoxes = app.cur_frame.bounding_boxes;
    if (boundingBoxes.length == 0) {
        return null;
    }
    var closestBox = null;
    var closestCorner = null;
    var shortestDistance = Number.POSITIVE_INFINITY;
    for (var i = 0; i < boundingBoxes.length; i++) {
        var b = boundingBoxes[i];
        var intersection = getIntersection(b);
        if (intersection) {
            if (intersection.distance < shortestDistance) {
                closestBox = b;
                closestCorner = intersection.point;
                shortestDistance = intersection.distance;
            }
        }
    }
    if (closestCorner) {
        return [closestBox, closestCorner];
    } else {
        return null;
    }
}

function intersectionWithCornerTopView(appBBOX){

    var closestBox = null;
    var closestCorner = null;
    var shortestDistance = Number.POSITIVE_INFINITY;
    var b = appBBOX;
    var intersection = getIntersectionTopView(b);
    if (intersection) {
        if (intersection.distance < shortestDistance) {
            closestBox = b;
            closestCorner = intersection.point;
            shortestDistance = intersection.distance;
        }
    }
    if (closestCorner) {
        return [closestBox, closestCorner];
    } else {
        return null;
    }
    
}


function getIntersectionTopView(b) {
    // var temp = new THREE.Vector3(mouse2DTopView.x, mouse2DTopView.y, 0);
    // temp.unproject( camera3 );
    // var dir = temp.sub( camera3.position ).normalize();
    // var distance = - camera3.position.y / dir.y;
    var pos  = getCurrentPositionCamera3(); // camera3.position.clone().add( dir.multiplyScalar( distance ) );
    // pos = {x: pos.z, y:0, z:pos.x}
    
    // var x = pos.z;
    // var y = pos.x;
    
    var shortestDistance = Number.POSITIVE_INFINITY;
    var closestCorner = null;
    
    for (var i = 0; i < b.geometry.vertices.length; i++) {
        if (distance2D(pos, b.geometry.vertices[i]) < shortestDistance &&
            distance2D(pos, b.geometry.vertices[i]) < b.get_cursor_distance_threshold()) {
            shortestDistance = distance2D(pos, b.geometry.vertices[i]);
            closestCorner = b.geometry.vertices[i];
        }
        
    }
    if (closestCorner == null) {
        return null;
    }
    
    return {distance: shortestDistance, point: closestCorner};
}

function getIntersection(b) {
    var temp = new THREE.Vector3(mouse2D.x, mouse2D.y, 0);
    temp.unproject( camera );
    var dir = temp.sub( camera.position ).normalize();
    var distance = - camera.position.y / dir.y;
    var pos = camera.position.clone().add( dir.multiplyScalar( distance ) );
    var shortestDistance = Number.POSITIVE_INFINITY;
    var closestCorner = null;
    for (var i = 0; i < b.geometry.vertices.length; i++) {
        if (distance2D(pos, b.geometry.vertices[i]) < shortestDistance &&
            distance2D(pos, b.geometry.vertices[i]) < b.get_cursor_distance_threshold()) {
            shortestDistance = distance2D(pos, b.geometry.vertices[i]);
            closestCorner = b.geometry.vertices[i];
        }
        
        
        
        // console.log("--pos", i, pos, b.geometry.vertices[i], distance2D(pos, b.geometry.vertices[i]) );
        
    }
    if (closestCorner == null) {
        return null;
    }
    
    // console.log("--closestCorner", closestCorner);
    return {distance: shortestDistance, point: closestCorner};
}

function distance2D(v1, v2) {
    return Math.pow(Math.pow(v1.x - v2.x, 2) + Math.pow(v1.z - v2.z, 2), 0.5)
}
  
function closestPoint(p, vertices) {
    var shortestDistance = Number.POSITIVE_INFINITY;
    var closestIdx = null;
    for (var i = 0; i < vertices.length; i++) {
        if (p.distanceTo(vertices[i]) < shortestDistance) {
            shortestDistance = p.distanceTo(vertices[i]);
            closestIdx = i;
        }
    }
    return closestIdx;
}

function save(boundingBoxes) {
  var outputBoxes = []
  for (var i = 0; i < boundingBoxes.length; i++) {
    outputBoxes.push(new OutputBox(boundingBoxes[i]));
  }
  var output = {"bounding boxes": outputBoxes};
  var stringifiedOutput = JSON.stringify(output);
  var file = new File([stringifiedOutput], "test.json", {type: "/json;charset=utf-8"});
  saveAs(file);
}

function save_image() {
    renderer.domElement.toBlob(function (blob) {
        saveAs(blob, "image.png");
    });
}

function upload_file() {
    var x = document.getElementById("file_input");
    if (x.files.length > 0) {
        reset();
        var file = x.files[0];
        load_text_file(file, import_annotations_from_bin);
        evaluator.resume_3D_time();
        evaluator.resume_time();
        $("#record").show();
        isRecording = true;
    }
}

var fileLoaded = true;
var currFile = "";
function upload_files() {
    $.ajax({
        url: '/loadFrameNames',
        type: 'POST',
        contentType: 'application/json;charset=UTF-8',
        success: function(response) {
            app.fnames = response.split(',');
            get_pointcloud_data(app.fnames[0]);
        },
        error: function(error) {
            console.log(error);
        }
    });
}



function load_data_helper(index, files) {
    if (index < evaluation.filenames.length) {
        load_text_file(index, files[index], files);
    }
}


function import_annotations_from_bin(data) {
  if ( data === '' || typeof(data) === 'undefined') {
    return;
  }
}


function load_text_file(index, text_file, files) {
  if (text_file) {
    var text_reader = new FileReader();
    text_reader.readAsArrayBuffer(text_file);
    text_reader.onload = function() {
        readData(text_reader);
        load_data_helper(index + 1, files);
    }
  }
}


// https://stackoverflow.com/a/15327425/4855984
String.prototype.format = function(){
    var a = this, b;
    for(b in arguments){
        a = a.replace(/%[a-z]/,arguments[b]);
    }
    return a; // Make chainable
};