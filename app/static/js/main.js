if (!Detector.webgl) Detector.addGetWebGLMessage();

var renderer, scene, camera, stats, raycaster, clock;
var mouse2D = new THREE.Vector2();
var mouse2DTopView = new THREE.Vector2();
var intersection = null;
var mouseDown;
var highlightMode = false;
var threshold = 0.5,
    pointSize = 1;

// data structures
var boundingBoxes = [],
    hoverBoxes = [];
var image_loaded = false;
var newBox, newBoundingBox, newBoxHelper;

var mouse = new THREE.Vector3(),
    anchor = new THREE.Vector3();
var mouseTopView = new THREE.Vector3();

var currentPosition = new THREE.Vector3();

var boxgeometry = new THREE.BoxGeometry(1, 1, 1);
var boxmaterial = new THREE.MeshDepthMaterial({
    opacity: .1
});
var selectedBox;
var indexedPoints;
var angle;
var hoverIdx, hoverBox;
var resizeBox, rotatingBox;
var isResizing = false;
var isMoving = false;
var isRotating = false;


var isResizingTopView = false;
var isMovingTopView = false;
var isRotatingTopView = false;


var grid;
var pointMaterial = new THREE.PointsMaterial({
    size: pointSize * 8,
    sizeAttenuation: false,
    vertexColors: THREE.VertexColors
});

var isRecording = true;
var app;
var mean, sd, filteredIntensities, min, max, intensities, colors;
var selected_color = new THREE.Color(0x78F5FF);
var hover_color = new THREE.Color(1, 0, 0);
var default_color = new THREE.Color(0xffff00);
var autoDrawMode = false;

init();


// called first, populates scene and initializes renderer
function init() {

    var container = document.getElementById('container');
    var panel3 = document.getElementById('panel3');
    var panelTopView = document.getElementById('panel');
    scene = new THREE.Scene();
    //scene.background = new THREE.Color( 0xfffeee );
    scene2 = new THREE.Scene();
    scene2.background = new THREE.Color(0x000000);

    scene3 = new THREE.Scene();
    scene3.background = new THREE.Color(0x000000);




    clock = new THREE.Clock();

    // set up PerspectiveCamera
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 1000);
    camera.position.set(0, 100, 0);
    camera.lookAt(new THREE.Vector3(0, 0, 0));


    camera2 = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 100000);
    camera2.position.set(-10, 0, 5);
    camera2.lookAt(new THREE.Vector3(0, 0, 0));


    camera3 = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 100000);
    camera3.position.set(0, 10, -5);
    camera3.lookAt(new THREE.Vector3(0, 0, 0));


    //
    grid = new THREE.GridHelper(200, 20, 0xffffff, 0xffffff);
    // scene.add( grid );

    // set up renderer
    renderer = new THREE.WebGLRenderer({
        preserveDrawingBuffer: true
    }, {
        antialias: true
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    //renderer.setSize(500,500);
    //renderer.setScissorTest( true );
    container.appendChild(renderer.domElement);

    //set up renderer2
    renderer2 = new THREE.WebGLRenderer({
        preserveDrawingBuffer: true
    }, {
        antialias: true
    });
    renderer2.setPixelRatio(window.devicePixelRatio);
    renderer2.setSize(500, 400);
    panel3.appendChild(renderer2.domElement);


    //set up renderer3
    renderer3 = new THREE.WebGLRenderer({
        preserveDrawingBuffer: true
    }, {
        antialias: true
    });
    renderer3.setPixelRatio(window.devicePixelRatio);
    renderer3.setSize(500, 400);
    panelTopView.appendChild(renderer3.domElement);


    //
    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = threshold;
    //Testing Parts
    var geometry = new THREE.IcosahedronGeometry(0.1);
    var material = new THREE.MeshLambertMaterial({
        color: 0xff0000
    });
    var cube = new THREE.Mesh(geometry, material);
    // 加入右边场景中
    // scene2.add(cube)
    var axes = new THREE.AxesHelper(20);
    var axes2 = new THREE.AxesHelper(20);
    //scene.add(axes);
    //scene2.add(axes2);
    var ambientLight = new THREE.AmbientLight(0x0c0c0c);
    scene.add(ambientLight);
    scene2.add(ambientLight);
    scene3.add(ambientLight);
    //
    stats = new Stats();
    //stats2 = new Stats();
    //stats3 = new Stats();
    container.appendChild(stats.dom);
    //panel3.appendChild(stats.dom);
    //panelTopView.appendChild(stats.dom);

    //
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls2 = new THREE.OrbitControls(camera2, renderer2.domElement);
    controls3 = new THREE.OrbitControls(camera3, renderer3.domElement);
    // controls.autoRotate = false;

    // camera.rotation.set(0, 0, 0.5 * 3.14); //final cam position
    // camera.position.set( 0, 20, 100 );
    // controls.update();

    //camera2.rotateZ(3.14);


    controls3.maxPolarAngle = 0;
    controls3.minPolarAngle = 0;
    camera3.updateProjectionMatrix();



    window.addEventListener('resize', onWindowResize, false);

    // document.getElementById('container').addEventListener( 'onclick', onDocumentClick, false );

    document.getElementById('panel').addEventListener('mousemove', onDocumentMouseMove, false);
    document.getElementById('panel').addEventListener('mousedown', onDocumentMouseDown, false);
    document.getElementById('panel').addEventListener('mouseup', onDocumentMouseUp, false);


    document.getElementById('panel').addEventListener('keydown', onKeyDown2, false);
    document.getElementById('panel').addEventListener('keyup', onKeyUp2, false);
    document.getElementById('panel').addEventListener('keypress', clickKeystrokeControl, false);




    document.getElementById('container').addEventListener('mousemove', onDocumentMouseMove, false);
    document.getElementById('container').addEventListener('mousedown', onDocumentMouseDown, false);
    document.getElementById('container').addEventListener('mouseup', onDocumentMouseUp, false);
    document.getElementById('container').addEventListener('keypress', clickKeystrokeControl, false);

    
    
    



    document.addEventListener('mousemove', updateMouse, false);
    document.getElementById('save').addEventListener('click', write_frame_out, false);
    document.getElementById('move').addEventListener('click', moveMode, false);
    document.getElementById('move2D').addEventListener('click', move2DMode, false);
    
    document.getElementById('objectIDs').addEventListener('click', objectListVisualization, false);
    
    document.addEventListener("keydown", onKeyDown2); //or however you are calling your method
    document.addEventListener("keyup", onKeyUp2);
    document.addEventListener("keypress", clickKeystrokeControl, false);
    // document.getElementById( 'record' ).addEventListener( 'click', toggleRecord, false );

    window.onbeforeunload = function(evt) {
        return true;
    }
    app = new App();
    app.init();

    enable_bounding_box_tracking = settingsControls['FrameTracking'];

    
    $("#panel2").html(`
        <div class="divTable">

           <div class="divTableBody">

              <div class="divTableRow">
                 <div class="divTableCell divNameCell">&nbsp;Object ID</div>
                 <div id="summary-object-id" class="valueCell divTableCell">&nbsp;</div>
              </div>

              <div class="divTableRow">
                 <div class="divTableCell divNameCell">&nbsp;Object Type</div>
                 <div id="summary-object-type" class="valueCell divTableCell">&nbsp;</div>
              </div>
              <div class="divTableRow">
                 <div class="divTableCell divNameCell">&nbsp;Size (w x h x l)</div>
                 <div id="summary-object-dimension" class="valueCell divTableCell">&nbsp;</div>
              </div>
<!--
              <div class="divTableRow">
                 <div class="divTableCell divNameCell">&nbsp;</div>
                 <div class="valueCell divTableCell">(width x length x height)</div>
              </div>
-->
              <div class="divTableRow">
                 <div class="divTableCell divNameCell">&nbsp;Object Angle</div>
                 <div id="summary-object-angle" class="valueCell divTableCell">&nbsp;</div>
              </div>
              <div class="divTableRow">
                 <div class="divTableCell divNameCell">&nbsp;Center Location</div>
                 <div id="summary-object-center-location" class="valueCell divTableCell">&nbsp;</div>
              </div>
     
              <div class="divTableRow">
                 <div class="divTableCell divNameCell">&nbsp;Is Auto-generated</div>
                 <div id="summary-object-isautogenrated" class="valueCell divTableCell">&nbsp;</div>
              </div>

                <div style="position: absolute;right: 7px;top: 90px;font-size: 12px;cursor: move;">&nbsp;<input  id="summary-object-islocked" type="checkbox" name="summary-object-isvalidated-input" value="1"> (L)ocked<br></div>


                <div id="recenter-objects" style="min-width: 62px; position: absolute;right: -6px;top: 111px;font-size: 12px;cursor: move;">&nbsp;<a href="#" ><i class="fa fa-dot-circle-o" aria-hidden="true"></i>&nbsp;&nbsp;Re-Cen(T)er</a></div>


                <div id="refresh-side-color" style="min-width: 62px; position: absolute;right: -6px;top: 139px;font-size: 12px;cursor: move;">&nbsp;<a href="#" ><i class="fa fa-refresh" aria-hidden="true"></i>&nbsp;&nbsp;R(E)-Colors</a></div>


           </div>
        </div>
    `);

    $("#summary-object-islocked").change(function(){
       
        toggle_locked_box(selectedBox);

    });
    
    $("#recenter-objects").click(function(){
        recenter_objects();
    });
    
    $("#refresh-side-color").click(function(){

        toogle_color();


    });

    $("#summary-object-islocked").parent().click(function(){
    
        var current_check = $("#summary-object-islocked").is(":checked");
        $("#summary-object-islocked").prop('checked', !current_check);

        toggle_locked_box(selectedBox);
    });
    
    // camera.rotateZ(3.14 * 0.5);

    $("#ClearObjectTable").click(function() {
        
         deleteAllBoundingBox(false);
    
    
    });


    $("#container").click(function() {

        var p = app.getCursor();

        for (var i_box = app.cur_frame.bounding_boxes.length - 1; i_box >= 0; i_box--) {
            box = app.cur_frame.bounding_boxes[i_box]
            //console.log("i_box", i_box);
            // if( bb_max.x >= p.x && bb_max.z >= p.z && bb_min.x <= p.x && bb_min.z <= p.z ) {

            if (p && containsPoint(box, p)) {


                selectedBox = box;
                app.bbox_visualization();
                return;
            } else {

                //console.log("not-deleted",p, box, app.cur_frame.bounding_boxes[i_box].id);
            }

        }

    });

}

function recenter_objects(){

        if(camera3.position.y < 6){
           selectedBox.innerRotate( Math.PI);
        }
        
        app.forceVisualize = true;
        app.bbox_visualization();
        app.forceVisualize = false;
    

}

function toogle_color(){
    
        if(app.isRedColor){
            recolor_evaluation();
            app.isRedColor = false;

        }else{
            //normalizeColors(app.cur_frame.data, null, app.annote_pointcloudXZ);
            app.isRedColor = true;
            app.forceVisualize = true;
            app.bbox_visualization();
            app.forceVisualize = false;
            
        
        }
}

function recolor_evaluation(){
    
    if(app.isRedColor && selectedBox){

        app.forceVisualize = true;
        app.bbox_visualization();
        app.forceVisualize = false;
        
       for ( var i = 0;  i < app.annote_pointcloudXZ.geometry.vertices.length; i ++ ) {

            var v =  app.annote_pointcloudXZ.geometry.vertices[i];
            var v = new THREE.Vector3( v.x, 0, v.z);
            if (v && containsPoint(app.selectedBox, v)) {
                app.annote_pointcloudXZ.geometry.colors[i].setRGB(0,255,0);
            }else{
                app.annote_pointcloudXZ.geometry.colors[i].setRGB(255,0,0);
            }

        }
       for ( var i = 0;  i < app.annote_pointcloudXZY.geometry.vertices.length; i ++ ) {

            var v =  app.annote_pointcloudXZY.geometry.vertices[i];
            var v = new THREE.Vector3( v.x, 0, v.z);
            if (v && containsPoint(app.selectedBox, v)) {
                app.annote_pointcloudXZY.geometry.colors[i].setRGB(0,255,0);
            }else{
                app.annote_pointcloudXZY.geometry.colors[i].setRGB(255,0,0);
            }

        }
        app.annote_pointcloudXZ.geometry.colorsNeedUpdate = true;
        app.annote_pointcloudXZ.material.size =  settingsControls.PointSize+0.5; 
        app.annote_pointcloudXZY.geometry.colorsNeedUpdate = true;
        app.annote_pointcloudXZY.material.size =  settingsControls.PointSize+0.5; 
    }
            
}

function toggle_locked_box(box){

        box.islocked = $("#summary-object-islocked").is(":checked");
        app.bbox_visualization();
        update_point_size();
    
        updateSelectOption(box);
    
        if(box.islocked){
            $("#summary-object-islocked").parent().css("color", "red");
        }else{
            $("#summary-object-islocked").parent().css("color", "white");
        }
        
    
      
}


function write_frame_out() {
    var FrameTracking = settingsControls["FrameTracking"];
    settingsControls["FrameTracking"] = true; // Force saving!!!
    app.write_frame_out();
    settingsControls["FrameTracking"] = FrameTracking;
}
// function write_frame() {
//     evaluator.pause_recording();
//     evaluation.add_evaluator(evaluator);
//     evaluation.write_frame();
// }

function predictLabel(boundingBox) {
    if (!enable_predict_label) {
        return;
    }
    if (boundingBox.hasPredictedLabel == false) {
        $.ajax({
            url: '/predictLabel',
            data: JSON.stringify({
                frames: [{
                    filename: app.cur_frame.fname,
                    bounding_boxes: [stringifyBoundingBoxes([boundingBox])[0]]
                }],
                filename: app.cur_frame.fname
            }),
            type: 'POST',
            contentType: 'application/json;charset=UTF-8',
            success: function(response) {
                var label = parseInt(response.split(",")[0], 10);
                var in_fov = response.split(",")[1] == "True";
                console.log(response, in_fov);
                boundingBox.hasPredictedLabel = true;
                if (label != -1) {
                    updateLabel(boundingBox.id, label);
                }
                /*
                if (in_fov) {
                    updateCroppedImagePanel('');
                } else {
                    updateCroppedImagePanel('outside FOV');
                }
                */

                $("#panel").hide();
                $("#panel").empty();

            },
            error: function(error) {
                console.log(error);
            }
        });
    }
}

function getMaskRCNNLabels(filename) {
    console.log("getMaskRCNNLabels");
    if (!enable_mask_rcnn) {
        return;
    }
    $.ajax({
        url: '/getMaskRCNNLabels',
        data: JSON.stringify({
            filename: filename
        }),
        type: 'POST',
        contentType: 'application/json;charset=UTF-8',
        success: function(response) {
            var l = response.length - 1;
            maskRCNNIndices = response.substring(1, l).split(',').map(Number);
            highlightPoints(maskRCNNIndices);
            updateMaskRCNNImagePanel();
        },
        error: function(error) {
            console.log(error);
        }
    });
}

function updateLabel(id, label) {
    var row = getRow(id);
    var dropDown = $(row).find("select");
    var selectedIndex = $(dropDown).prop("selectedIndex");
    $(dropDown).prop("selectedIndex", label);
    // evaluator.decrement_label_count();
    // app.f
}

// gets 2D mouse coordinates
function updateMouse(event) {
    event.preventDefault();
    mouse2D.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse2D.y = -(event.clientY / window.innerHeight) * 2 + 1;

    //$("#footer-top-view").html(getCaretPosition($('#panel')));

    if (isInsideTheVisBox(event)) {

        var panelMouse = document.getElementById("panel").getBoundingClientRect();
        mouse2DTopView.x = ((event.clientX - panelMouse.x) / panelMouse.width) * 2 - 1;
        mouse2DTopView.y = -((event.clientY - panelMouse.y) / panelMouse.height) * 2 + 1;

    } else {

        mouse2DTopView.x = -999;
        mouse2DTopView.y = -999;
    }

    // console.log(event.clientX , event.clientY, mouse2D, mouse2DTopView);
}



// controller for resizing, rotating, translating, or hovering boxes and points
function onDocumentMouseMove(event) {
    event.preventDefault();

    if (!isRecording) {
        return;
    }

    
    
    if (isInsideTheVisBox(event)) {

        var cursor = getCurrentPositionCamera3();
        changeCursor(isMovingTopView, cursor, app.selectedBox);

        if (!controls.enabled) {


            highlightCornersTopView();

            
            if(isMovingTopView== false){
                app.selectedBox.changeBoundingBoxColor(hover_color.clone());
            }
 
            
            
            cursor.y -= app.eps;

            app.not_update_all_bbox = true;
            if (mouseDown && selectedBox.islocked==false) {
                
                

                if (isRotatingTopView) {


                    var oldAngle = app.selectedBox.angle;
                    app.selectedBox.rotate(cursor);
                    app.selectedBox.add_timestamp();

                    selectedBox.innerRotate(app.selectedBox.angle - oldAngle);
                    selectedBox.add_timestamp();


                } else if (isResizingTopView) {



                    // var cursor = new THREE.Vector3( cursor.x, cursor.z,cursor.y );

                    app.selectedBox.resize(cursor);
                    app.selectedBox.add_timestamp();

                    // Updating to Main-Scene

                    var cursorMain = selectedBox.initialcursor.clone();
                    var cursorUpdate = app.selectedBox.changesOnLatestResize(cursor);

                    cursorMain.x = cursorMain.x + (cursorUpdate.x * 1) //(panelMouse.height / window.innerHeight));
                    cursorMain.y = cursorMain.y + cursorUpdate.y;
                    cursorMain.z = cursorMain.z + (cursorUpdate.z * 1) //  (panelMouse.width / window.innerWidth) )  ;

                    selectedBox.resize(cursorMain);
                    selectedBox.add_timestamp();



                } else if (isMovingTopView && selectedBox.initialcursor) {

                  
                             
                            app.selectedBox.translate(cursor);
                            app.selectedBox.changeBoundingBoxColor(new THREE.Color(0, 1, 1));
                            app.selectedBox.add_timestamp();
                             

                            var cursorMain = selectedBox.initialcursor.clone();
                            var cursorUpdate = app.selectedBox.changesOnLatestResize(cursor); 

                            cursorMain.x = cursorMain.x + (cursorUpdate.x * 1) //(panelMouse.height / window.innerHeight));
                            cursorMain.y = cursorMain.y + cursorUpdate.y ;
                            cursorMain.z = cursorMain.z + (cursorUpdate.z * 1) //  (panelMouse.width / window.innerWidth) )  ;

                            selectedBox.translate(cursorMain);
                            selectedBox.changeBoundingBoxColor(selected_color.clone());
                            selectedBox.add_timestamp();
                    
                            
                        

                }


                app.bbox_visualization();

                app.not_update_all_bbox = false;

            } else {

                app.bbox_visualization();

                predictLabel(app.selectedBox);

                app.not_update_all_bbox = false;

            }
            
            
            
        }


    } else {


        $("body").css("cursor", "default");
        if(selectedBox && selectedBox.islocked==true){
            return;
        }

        app.handleBoxRotation();
        app.handleBoxResize();
        app.handleBoxMove();



        if (mouseDown && !isRotating && !isResizing && !isMoving &&
            !isRotatingTopView && !isMovingTopView && !isResizingTopView
        ) {
            if (newBox != null && !newBox.added) {
                scene.add(newBox.points);
                scene.add(newBox.boxHelper);
                newBox.added = true;
            }
            if (newBox) {
                newBox.resize(app.getCursor());
            }
        }

        var cursor = getCurrentPosition();
        if (!controls.enabled) {
            // console.log("controls not enabled");
            // highlights all hover boxes that intersect with cursor
            updateHoverBoxes(cursor);


            highlightCorners();
            app.bbox_visualization();
        }

    }


}


// updates hover boxes and changes their colors to blue
function updateHoverBoxes(v) {
    var boundingBoxes = app.cur_frame.bounding_boxes;
    if (!isMoving) {
        hoverBoxes = [];
        for (var i = 0; i < boundingBoxes.length; i++) {
            var box = boundingBoxes[i];
            // added box to boverBoxes if cursor is within bounding box
            if (v && containsPoint(box, v)) {
                hoverBoxes.push(box);
            }

            // checks if box is selectedBox, if so changes color back to default
            if (box != selectedBox) {
                box.changeBoundingBoxColor(default_color.clone());
            }
        }

        // update color of hover box if only one box is hovered
        if (hoverBoxes.length == 1) {
            var box = hoverBoxes[0];
            if (box != selectedBox) {
                box.changeBoundingBoxColor(hover_color.clone());
            }
        }
    }
}




var camera_angle;

// controller for adding box
function onDocumentMouseUp(event) {
    event.preventDefault();
    if (!isRecording) {
        return;
    }
    if (isRecording) {
        app.handleAutoDraw();


        mouseDown = false;
        var predictBox = null;
        if (newBox != null && newBox.added) {
            addBox(newBox);
            newBox.add_timestamp();
            app.increment_add_box_count();
            predictBox = newBox;

            selectedBox = newBox;
            app.bbox_visualization();
        }
        newBox = null;
        if (isResizing) {
            app.increment_resize_count();
            predictLabel(resizeBox);
            predictBox = resizeBox;
        }
        if (isMoving && selectedBox) {
            app.increment_translate_count();
            predictLabel(selectedBox);
            predictBox = selectedBox;
        }
        if (isRotating) {
            app.increment_rotate_count();
            predictBox = rotatingBox;
        }
        if (predictBox) {
            predictLabel(predictBox);
        }
        isResizing = false;
        isRotating = false;
        // if (isMoving) {
        //     changeBoundingBoxColor(hoverBoxes[0], new THREE.Color( 7,0,0 ));
        // }
        isMoving = false;

        isRotatingTopView = false;
        isMovingTopView = false;
        isResizingTopView = false;

        highlightCornersTopView();


        // if (app.move2D) {
        app.increment_rotate_camera_count(camera.rotation.z);
        // }
    }
}

function onDocumentMouseDown(event) {

    event.preventDefault();


    if (!isRecording) {
        return;
    }

    var isFirstClickInsidetheVisBox = false;

    if (isInsideTheVisBox(event)) {
        isFirstClickInsidetheVisBox = true;
    }
    
    
    $("body").css("cursor", "default");
    if (!controls.enabled) {
        mouseDown = true;

        if (isInsideTheVisBox(event)) {
            isFirstClickInsidetheVisBox = true;

            
            var pos = getCurrentPositionCamera3(); // get2DCoord();

            
            app.selectedBox.changeBoundingBoxColor(0xffff00);
            var intersection = intersectionWithCornerTopView(app.selectedBox);

            app.selectedBox.changeBoundingBoxColor(0xffff00);

            //console.log("onDocumentMouseDown", pos, containsPoint(app.selectedBox, pos), intersection);
            

            var FrameCursor = selectedBox.get_center().clone();
            var mainFrameCursor = new THREE.Vector3(FrameCursor.y, 0.0001, FrameCursor.x);
            
            mainFrameCursor.x = mainFrameCursor.x + pos.x;
            mainFrameCursor.z = mainFrameCursor.z + pos.z;

            if (intersection != null) {
                var box = app.selectedBox; // intersection[0];

                var anchor = intersection[1];
                var closestIdx = closestPoint(anchor, box.geometry.vertices);

                if (closestIdx == 4) {
                    isRotatingTopView = true;
                } else {
                    isResizingTopView = true;

                }

                box.anchor = box.geometry.vertices[getOppositeCorner(closestIdx)].clone();
                box.initialcursor = box.geometry.vertices[closestIdx].clone();

                selectedBox.anchor = selectedBox.geometry.vertices[getOppositeCorner(closestIdx)].clone();
                selectedBox.initialcursor = selectedBox.geometry.vertices[closestIdx].clone();
                //selectedBox.initialcursor = mainFrameCursor.clone();


            } else if (pos && containsPoint(app.selectedBox, pos)) { // Is hovering

                isMovingTopView = true;
                app.selectedBox.changeBoundingBoxColor(new THREE.Color(0, 1, 1));
                app.selectedBox.cursor =  pos.clone();
                
                app.selectedBox.initialcursor =  pos.clone();
                selectedBox.cursor = mainFrameCursor.clone();
                selectedBox.initialcursor = mainFrameCursor.clone();

            } else if (pos && containsPoint(app.selectedBox, pos) == false) {
                isMovingTopView = false;

            }

        } else {
            
            
            anchor = get3DCoord();
            var intersection = intersectWithCorner();
            // update hover box
            if (selectedBox && (hoverBoxes.length == 0 || hoverBoxes[0] != selectedBox)) {
                selectedBox.changeBoundingBoxColor(0xffff00);
                selectedBox = null;
                isMoving = false;
            }

            if (intersection != null) {
                var box = intersection[0];
                var closestIdx = closestPoint(anchor, box.geometry.vertices);
                // console.log("closest: ", closestIdx);

                if (closestIdx == 4) {
                    isRotating = true;
                    rotatingBox = box;
                } else {
                    isResizing = true;
                    resizeBox = box;
                    resizeBox.anchor = resizeBox.geometry.vertices[getOppositeCorner(closestIdx)].clone();
                }
            } else if (hoverBoxes.length == 1) {
                isMoving = true;
                hoverBoxes[0].select(get3DCoord());
                selectRow(selectedBox.id);

            } else if (isFirstClickInsidetheVisBox == false) {
                angle = camera.rotation.z;
                var v = anchor.clone();
                anchor.x += .000001;
                anchor.y -= .000001;
                anchor.z += .000001;  
                newBoundingBox = new THREE.Box3(anchor, v);
                newBoxHelper = new THREE.Box3Helper(newBoundingBox, 0xffff00);
                anchor = anchor.clone();

                newBox = new Box(anchor, v, angle, newBoundingBox, newBoxHelper);

            }

        }
    }
}


function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);

}

function animate2() {


    requestAnimationFrame(animate2);
    render2();
    render3();
   
    //stats.update();
    //stats3.update();

}

function animate() {

    requestAnimationFrame(animate);

    render();
    stats.update();
    //stats2.update();

}

/* Gets the 3D cursor position that is projected onto the z plane */
function getCurrentPosition() {
    var temp = new THREE.Vector3(mouse2D.x, mouse2D.y, 0);
    temp.unproject(camera);
    var dir = temp.sub(camera.position).normalize();
    var distance = -camera.position.y / dir.y;
    var pos = camera.position.clone().add(dir.multiplyScalar(distance));
    return pos;
}


function getCurrentPositionCamera3() {


    var temp = new THREE.Vector3(mouse2DTopView.x, mouse2DTopView.y, 0);
    temp.unproject(camera3);
    var dir = temp.sub(camera3.position).normalize();
    var distance = -camera3.position.y / dir.y;
    var pos = camera3.position.clone().add(dir.multiplyScalar(distance));
    return pos;
}



var toggle = 0;

function render2() {
    toggle += clock.getDelta();
    renderer2.render(scene2, camera2);
}

function render3() {
    toggle += clock.getDelta();
    renderer3.render(scene3, camera3);
    update_footer_camera3();
}

function render() {
    toggle += clock.getDelta();
    //renderer.render( scene, camera );
    //renderer.setScissor( 0, 0, window.innerWidth/2 , window.innerHeight );
    renderer.render(scene, camera);
    //renderer2.render(scene2, camera2);
    //renderer3.render(scene3, camera3);
    //renderer.setScissor( window.innerWidth / 2 , 0, window.innerWidth / 2, window.innerHeight );
    //renderer.render( scene2, camera2 );
    //controls.update();

    app.render_text_labels();

    if (app.move2D) {
        grid.rotation.y = camera.rotation.z;
    }
    update_footer(getCurrentPosition());

    update_footer_camera3();
}

function update_footer_camera3() {



    var pos = getCurrentPositionCamera3();
    var y = pos.z;
    var x = pos.x;

    $("#footer-top-view").find("p").html("x: {0}{1}y: {2}".format(x.toFixed(3),
        "<br />",
        y.toFixed(3)));

    if (mouse2DTopView.x == -999 && mouse2DTopView.y == -999) {

        $("#footer-top-view").hide();
    } else {

        $("#footer-top-view").show();
    }

}

function update_footer(pos) {
    
    var reminder_text = "";
    if (isRecording) {
        if (app.move2D) {
            if (controls.enabled == true) {
                reminder_text = "Hold control key and click on point cloud to start drawing bounding box";
            } else {
                if (isResizing) {
                    reminder_text = "Release mouse to stop resizing box";
                } else if (isMoving) {
                    reminder_text = "Release mouse to stop translating box";
                } else if (isRotating) {
                    reminder_text = "Release mouse to stop rotating box";
                } else if (mouseDown) {
                    reminder_text = "Release mouse to stop drawing box";
                } else {
                    reminder_text = "Click on point cloud to start drawing bounding box"
                }
            }
        }
    } else {
        reminder_text = "Resume recording to continue annotating";
    }

    $("#draw_bounding_box_reminder").find("p").text(reminder_text);
    // console.log(reminder_text);


    var x = pos.z;
    var y = pos.x;

    $("#footer").find("p").html("x: {0}{1}y: {2}".format(x.toFixed(3),
        "<br />",
        y.toFixed(3)));
}



function generatePointCloud() {
    if (app.cur_pointcloud != null) {
        return updatePointCloud(app.cur_frame.data, COLOR_RED);
    } else {
        return generateNewPointCloud(app.cur_frame.data, COLOR_RED, true);
    }
}


function switchMoveMode() {
    eventFire(document.getElementById('move'), 'click');
}

function switch2DMode() {
    eventFire(document.getElementById('move2D'), 'click');
}

function moveMode(event) {
    event.preventDefault();
    // assertRecordMode();
    $("#object-table").hide();
    $("#frames-table").show();
    if (isRecording) {
        controls.enabled = true;
        app.move2D = false;
        document.getElementById('move2D').className = "";
        document.getElementById('objectIDs').className = "";
        document.getElementById('move').className = "selected";
        controls.maxPolarAngle = 2 * Math.PI;
        controls.minPolarAngle = -2 * Math.PI;
        app.resume_3D_time();
    }
    unprojectFromXZ();
}

// function assertRecordMode() {
//     if (!isRecording) {
//         alert("Resume recording to change modes");
//     }
// }
// function select2DMode() {
//     console.log("draw");
//     document.getElementById( 'move' ).className = "";
//     document.getElementById( 'move2D' ).className = "selected";
//     camera.position.set(0, 100, 0);
//     camera.lookAt(new THREE.Vector3(0,0,0));
//     // camera.rotation.y = 0;
//     controls.maxPolarAngle = 0;
//     controls.minPolarAngle = 0;
//     camera.updateProjectionMatrix();
//     projectOntoXZ();

//     controls.reset();
//     controls.enabled = true;
//     controls.update();
//     app.move2D = true;
// }

function objectListVisualization(event){

    event.preventDefault();
    
    document.getElementById('move').className = "";
    document.getElementById('move2D').className = "";    
    document.getElementById('objectIDs').className = "selected";
    
    $("#object-table").show();
    $("#frames-table").hide();
    

}

function move2DMode(event) {
    event.preventDefault();
    $("#object-table").hide();
    $("#frames-table").show();
    if (isRecording) {
        document.getElementById('move').className = "";
        document.getElementById('objectIDs').className = "";
        document.getElementById('move2D').className = "selected";
        if (!app.move2D) {
            controls.maxPolarAngle = 0;
            controls.minPolarAngle = 0;
            camera.updateProjectionMatrix();
            projectOntoXZ();
            // controls.reset();

            app.pause_3D_time();
        }
        controls.enabled = true;
        controls.update();
        app.move2D = true;
    }

}

function projectOntoXZ() {
    var count = 0;
    var colors = app.cur_pointcloud.geometry.colors;
    for (var i = 0; i < app.cur_pointcloud.geometry.vertices.length; i++) {
        var v = app.cur_pointcloud.geometry.vertices[i];
        if (colors[i].b > colors[i].r) {
            count += 1;
            v.y = -0.001;
        } else {
            v.y = 0;
        }
    }
    app.cur_pointcloud.geometry.verticesNeedUpdate = true;
}

function unprojectFromXZ() {
    if (app.cur_frame) {
        for (var i = 0; i < app.cur_pointcloud.geometry.vertices.length; i++) {
            var v = app.cur_pointcloud.geometry.vertices[i];
            v.y = app.cur_frame.ys[i];
        }
        app.cur_pointcloud.geometry.verticesNeedUpdate = true;
    }
}


function reset() {
    var boundingBoxes = app.cur_frame.bounding_boxes;
    if (boundingBoxes) {
        for (var i = 0; i < boundingBoxes.length; i++) {
            box = boundingBoxes[i];
            scene.remove(box.boxHelper);
            scene.remove(box.points);
            clearTable();
        }
        boundingBoxes = [];
        yCoords = null;
        yCoords = [];
    }
}