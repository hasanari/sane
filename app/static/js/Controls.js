var maxSize = 4;
var pointSize = 2;
var old_camera = null;

var SettingsControls = function() {
    this.Size = pointSize / maxSize;
    this.OutlierRemoval = 'None';
    this.Clustering = 'DBSCAN';
    this.FittingCriterion = 1;
    this.WithXCRF = false;
    this.FrameTracking = true;
    this.WithDenoising = true;
    this.AutoDeleteExistingBbox = true;
    this.speed = 0.8;
    this.SamplingRate = 0.5;

    this.PointSize = 1.0;
    this.FullyAutomatedBbox = false;
    this.AutoUpdate = true;

    this.SearchRange = 4.0;
    this.NeighborsRadius = 8;

    this.GuidedTracking = true;
    this.PaddingSize = 0.2;
    this.AxisSlidingNumbers = 25;
    this.TolerancePoint = 25;
    
    
    this.AnnotatorId = 'Guest';
    this.GroundRemoval = false;
    this.ShapeFitting = false;

    this.ShowGrid = false;
    this.ActiveLearning = false;
    this.explode = function() {};
};


var gui = new dat.GUI({
    autoPlace: false
});

gui.domElement.id = 'gui-control';
$('.moveGUI').append($(gui.domElement));

var settingsControls = new SettingsControls();
var settingsFolder = gui //.addFolder('settings');

var annId = settingsFolder.add(settingsControls, 'AnnotatorId', ['Guest', 'Kim-Ji-Eun', 'GroundTruth', 'Hasan', 'Akhil', 'Zuxin', 'Mansur', 'Guilin', 'Manoj', 'Runzhong']).onChange(function() {
    // Load Annotation - based on user ID

    if (app && app.cur_frame) {

        var annotatorID = settingsControls["AnnotatorId"];
        settingsControls["AnnotatorId"] = app.annotatorID;
        app.write_frame_out();
        app.annotatorID = annotatorID;
        settingsControls["AnnotatorId"] = app.annotatorID;


        $("#annId select").prop("disabled", "disabled");
        app.lock_frame = false;
        var fname = app.cur_frame.fname;
        var framelist = Object.keys(app.frames);


        app.bbox_visualization_clearance();
        if (app.cur_frame) {

            for (var i_box = app.cur_frame.bounding_boxes.length - 1; i_box >= 0; i_box--) {
                box = app.cur_frame.bounding_boxes[i_box];
                if(box){
                    delete_one_box(box);
                }
                

            }

            updateAllObjectIds();

            updateCountBBOX();

            app.cur_frame.bounding_boxes = [];
            delete app.cur_frame;
        }



        for (var i = 0; i < framelist.length; i++) {
            var bboxes = app.frames[framelist[i]].bounding_boxes;


            for (var j = bboxes.length - 1; j >= 0; j--) {


                var box = bboxes[j];




                delete bboxes[j];
            }
            app.frames[framelist[i]].bounding_boxes = [];
            delete app.frames[framelist[i]];
        }

        app.cur_frame = null;
        app.set_frame(fname);
        console.log("set_frame", settingsControls["AnnotatorId"]);

    }
});

annId.domElement.id = "annId";

settingsFolder.add(settingsControls, 'FullyAutomatedBbox').onChange(function() {



    app.cur_frame.annotated = false;

    if (settingsControls['FullyAutomatedBbox'] == true) {


        app.frame_lock = true;
        fname = app.cur_frame["fname"]
        //app.frames = []
        app.fully_automated_bbox(fname);
        app.frame_lock = false;


    }
    updateCountBBOX();
    
    

});




settingsFolder.add(settingsControls, 'FittingCriterion', {
    '(A)rea': 0,
    '(C)loseness': 1,
    '(V)ariance': 3
});

settingsFolder.add(settingsControls, 'FrameTracking').onChange(function() {

    enable_bounding_box_tracking = settingsControls['FrameTracking'];
    
    if (enable_bounding_box_tracking) {
        $("#GuidedTracking").parent().parent().show();
    } else {
        $("#GuidedTracking").parent().parent().hide();
    }
    
});

GuidedTracking = settingsFolder.add(settingsControls, 'GuidedTracking');
GuidedTracking.domElement.id = "GuidedTracking";



var PointsFolder = gui.addFolder('Points');

PointsFolder.add(settingsControls, 'Clustering', {
    '(D)BSCAN': 'DBSCAN',
    'Region (G)rowing': 'OriginAwareClustering'
}).onChange(function() {
    if (settingsControls["Clustering"] == "OriginAwareClustering") {
        $("#SearchRange").parent().parent().show();
    } else {
        $("#SearchRange").parent().parent().hide();
    }
});

var SearchRange = PointsFolder.add(settingsControls, 'SearchRange').min(1.0).max(5.0).step(0.05)
SearchRange.domElement.id = "SearchRange";



PointsFolder.add(settingsControls, 'OutlierRemoval', ['None', 'RANSAC', 'PCA']).onChange(function() {
    if (settingsControls["OutlierRemoval"] == "RANSAC") {
        $("#samplingRate").parent().parent().show();
    } else {
        $("#samplingRate").parent().parent().hide();
    }
});
var samplingRate = PointsFolder.add(settingsControls, 'SamplingRate').min(0.1).max(1.0).step(0.05)
samplingRate.domElement.id = "samplingRate";
$("#samplingRate").parent().parent().hide();


PointsFolder.add(settingsControls, 'GroundRemoval');
PointsFolder.add(settingsControls, 'WithDenoising').onChange(function() {


    backupBBOX = app.cur_frame.bounding_boxes;


    app.frame_lock = false;
    fname = app.cur_frame["fname"]
    app.frames = []

    app.tempBBOX = [] // backupBBOX;
    app.set_frame(fname);

    updateCountBBOX();

});




// settingsFolder.add(settingsControls, 'FittingCriterion', { '(a) Area': 0, '(s) Closeness': 1,  '(d) PointCNN': 2, '(f) Variance': 3} );


var MiscFolder = gui.addFolder('Misc');

MiscFolder.add(settingsControls, 'ActiveLearning');
MiscFolder.add(settingsControls, 'AutoDeleteExistingBbox');
MiscFolder.add(settingsControls, 'ShapeFitting');


var visualizeFolder = gui.addFolder('Display');

visualizeFolder.add(settingsControls, 'NeighborsRadius').min(0.0).max(20.0).step(0.05).onChange(function() {
    app.bbox_visualization();
});



var size = 5000;
var divisions = 500;

var grid_helper = new THREE.GridHelper(size, divisions);
grid_helper.name = "grid_helper"

visualizeFolder.add(settingsControls, 'ShowGrid').onChange(function() {


    if (settingsControls["ShowGrid"]) {

        scene.add(grid_helper);
    } else {

        var selectedObject = scene.getObjectByName("grid_helper");
        scene.remove(selectedObject);


        var selectedObject = scene.getObjectByName("grid_helper");
        scene2.remove(selectedObject);


        var selectedObject = scene.getObjectByName("grid_helper");
        scene3.remove(selectedObject);


    }


    animate();

    app.bbox_visualization();

});




visualizeFolder.add(settingsControls, 'PointSize').min(0.0).max(10.0).step(0.01).onChange(function() {
    app.cur_pointcloud.material.size = settingsControls.PointSize; // * maxSize;
    pointMaterial.size = 3 + 2 * settingsControls.PointSize;
    app.bbox_visualization();
});

// gui.remember(SettingsControls);

PointsFolder.open();
settingsFolder.open();

$(".property-name").each(function(index) {
    var textName = $(this).text();
    $(this).text(textName.replace(/([A-Z])/g, ' $1').trim())

});


function toggleRecord() {
    // pause recording
    if (isRecording) {
        $("#record").text("Click here to resume recording");
        app.pause_recording();
        // move2DMode(event);
        isRecording = false;
        controls.enabled = false;
        controls.update();

    } else {
        // resume recording
        isRecording = true;
        $("#record").text("Click here to pause recording");
        app.resume_recording();

        controls.enabled = true;
        controls.update();
    }
}


// controller for pressing hotkeys
function clickKeystrokeControl(event) {

    var KeyID = event.keyCode;
    //console.log("KeyID", KeyID);
    switch (KeyID) {

        case 68:
        default:
            break;
    }



}



// controller for pressing hotkeys
function onKeyDown2(event) {
    if (isRecording) {
        
        var old_camera = camera.rotation.clone();
        var epsilon = 0.05;
        var KeyID = event.keyCode;
        
        
        if (event.ctrlKey) {
            toggleControl(false);

        }
        
        if(KeyID == 16){ // shiftkey pressed
            isShiftPressed = true;
        }
        
        
        //console.log("KeyID", KeyID);
        switch (KeyID) {
            case 8: // backspace
                deleteSelectedBox();
                break;
                
                
            case 37: // left key
                moveBoxLocations(0, -epsilon, false);
                break;
                
            case 38: // up key
                moveBoxLocations(-epsilon, 0, false);
                break;
                
            case 39: // right key
                moveBoxLocations(0, epsilon, false);
                break;
                
            case 40: // bottom key
                moveBoxLocations(epsilon, 0, false);
                break;
                
                
                
                
            case 46: // delete
                deleteSelectedBox();
                break;


            case 65: // a key
                settingsControls.FittingCriterion = 0;
                autoDrawModeToggle(true);
                break;
            case 67: // c key
                settingsControls.FittingCriterion = 1;
                autoDrawModeToggle(true);
                break;


            case 86: // v key
                settingsControls.FittingCriterion = 3;
                autoDrawModeToggle(true);
                break;


            case 90: // z key
                showPreviousFrameBoundingBoxToggle(true);
                break;
                
                
                
            case 188: // < key
                  current_shift = isShiftPressed;
                  isShiftPressed = false;
                  moveBoxLocations(0, -epsilon, false);
                
                  isShiftPressed = current_shift;
                break;
            case 190: // > key
                
                 current_shift = isShiftPressed;
                 isShiftPressed = false;
                 moveBoxLocations(0, epsilon, false);
                
                 isShiftPressed = current_shift;
                break;
                

            case 68:
            default:
                break;
        }

        settingsFolder.updateDisplay();
    }

}

// controller for releasing hotkeys
function onKeyUp2(event) {

    if (isRecording) {
        var KeyID = event.keyCode;

        
        if(KeyID == 16){ // shiftkey pressed
            isShiftPressed = false;
        }
        
        toggleControl(true);
        if (KeyID == 17) { // shiftkey ctrl key


             app.bbox_visualization();
             
            //app.bbox_visualization();
        }
        switch (KeyID) {

            case 32: // space key
                gotonextFrame();
                break;
                
                
           
            case 82: // r key
                toogle_color();
                if(app.move2D){
                    camera3.rotation.setFromVector3(camera.rotation.clone());  
                }
                break;
                
            case 65: // a key
                autoDrawModeToggle(false);
                break;
            case 67: // c key
                autoDrawModeToggle(false);
                break;
            case 68: // d key
                settingsControls.Clustering = "DBSCAN";
                
                PointsFolder.updateDisplay();
                autoDrawModeToggle(false);
                break;
                
            case 69: // e key

                if (selectedBox) {

                    var current_check = $("#summary-object-islocked").is(":checked");
                    $("#summary-object-islocked").prop('checked', !current_check);

                    toggle_locked_box(selectedBox);

                    if(app.move2D){
                        camera3.rotation.setFromVector3(camera.rotation.clone());  
                    }      
                    
                }


                break;
                

            case 71: // g key
                settingsControls.Clustering = "OriginAwareClustering";
                
                PointsFolder.updateDisplay();
                autoDrawModeToggle(false);
                break;

                
                
            case 81: // q key
                gotopreviousObject();
                break;

                
            case 84: // t key
                recenter_objects();
                if(app.move2D){
                    camera3.rotation.setFromVector3(camera.rotation.clone());  
                }      
                break;
                
                
            case 86: // v key
                autoDrawModeToggle(false);
                break;



            case 87: // w key
                gotonextObject();
                break;

            default:
                toggleControl(true);
                break;
        }
        
        
        settingsFolder.updateDisplay();
    }

    
    controls.update();
    
    if (KeyID == 17) { // shiftkey ctrl key


        toggleControl(true);
        
        if(app.move2D){
            camera3.rotation.setFromVector3(camera.rotation.clone());  
        }  
        

    }
    
    hoverBoxes.pop()

    
}

function is_all_objects_locked(){


    if (app.cur_frame){
         for (var i_box = app.cur_frame.bounding_boxes.length - 1; i_box >= 0; i_box--) {
             if( app.cur_frame.bounding_boxes[i_box].islocked ==false ){
                 
                box = getBoxById(app.cur_frame.bounding_boxes[i_box].id);
                if (box) {


                    app.forceVisualize = true;
                    selectedBox = box;
                    app.bbox_visualization();

                    app.forceVisualize = false;

                }
                 
                 return false;
             }
             
         }
    }
    return true;
} 

function gotonextFrame(){
    if (app.cur_frame){
        
         var frame_idx = app.get_frame_idx(app.cur_frame.fname);
        if(is_all_objects_locked()){

            clearObjectTable();
            
            $("#objectIDs").html('<i class="fa fa-list"></i>&nbsp;&nbsp;IDs (0)');
            app.bbox_visualization_clearance();


            unfocus_frame_row(getFrameRow(app.fnames[frame_idx]));
            focus_frame_row(getFrameRow(app.fnames[frame_idx+1]));
            
            app.set_frame(app.fnames[frame_idx+1]);
       

        }else{
            alert("Please locked all the boxes before go to the next frame!");
            focus_frame_row(getFrameRow(app.fnames[frame_idx]));
        }
    }
}

function deleteFrameByFname(fname){

   if(fname in app.frames){
   
       var cur_frame = app.frames[fname];
   
         for (var i_box = cur_frame.bounding_boxes.length - 1; i_box >= 0; i_box--) {
             box =cur_frame.bounding_boxes[i_box]
             delete_one_box(box);

        }

        delete app.frames[fname].data; 
        delete app.frames[fname]; 
       
       console.log("delete fname", fname);
   }
}

function ReloadCurrentFrame(){
    if (app.cur_frame){
        $(OBJECT_TABLE).find('tbody tr').hide();
        clearObjectTable();
        app.bbox_visualization_clearance();
        
        
        var fname = app.cur_frame.fname;
        deleteFrameByFname(fname);
      
        app.cur_frame = null;         
        app.lock_frame = false;    
        app.set_frame(fname);
        
    }
    
    return false;
}

function gotonextObject() {
    return gottoObject(1);
}

function gotopreviousObject() {

    return gottoObject(-1);
}

function gottoObject(object_location) {

    if (app.cur_frame &&  app.cur_frame.bounding_boxes.length > 0 && controls.enabled) {
        
        if(selectedBox){
            
            var box_ids =[];

            var current_box_idx = selectedBox.id;
            for (var i = 0; i < app.cur_frame.bounding_boxes.length; i++) {
                box_ids.push(app.cur_frame.bounding_boxes[i].id);
                if (selectedBox.id == app.cur_frame.bounding_boxes[i].id) {
                    current_box_idx = i;
                }
            }

            if(current_box_idx + object_location < 0){
                object_location = box_ids.length  - current_box_idx -1 ;
            }

            var boxId = box_ids[ (current_box_idx + object_location) % box_ids.length ];
        }else{
            var boxId =  app.cur_frame.bounding_boxes[0].id;
            
        }
        box = getBoxById(boxId);
        if (box) {


            app.forceVisualize = true;
            selectedBox = box;
            app.bbox_visualization();

            app.forceVisualize = false;
            

        }
    }

    return null;

}

function showPreviousFrameBoundingBoxToggle(b) {
    app.show_previous_frame_bounding_box();
}

function autoDrawModeToggle(b) {
    settingsFolder.updateDisplay();
    autoDrawMode = b;
}


function updateMaskRCNNImagePanel() {
    return;
    $("#panel").empty();
    $("#panel").prepend('<img src="static/images/masked_image.jpg" />');
    $("#panel").find("img").attr({
        'src': "static/images/masked_image.jpg?foo=" + new Date().getTime()
    });
    $("#panel").slideDown("slow");
}

function updateCroppedImagePanel(child) {
    $("#panel2").empty();
    if (child == 'outside FOV') {
        $("#panel2").prepend("Bounding box is outside camera's FOV");
    } else {
        $("#panel2").prepend('<img src="static/images/cropped_image.jpg" />');
        $("#panel2").find("img").attr({
            'src': "static/images/cropped_image.jpg?foo=" + new Date().getTime()
        });
        $("#panel2").slideDown("slow");
    }

}

// controller for pressing hotkeys
function onKeyDown(event) {
    if (isRecording) {
        if (event.ctrlKey) {
            toggleControl(false);

        }
        var KeyID = event.keyCode;
        switch (KeyID) {
            case 8: // backspace
                deleteSelectedBox();
                break;
            case 46: // delete
                deleteSelectedBox();
                break;
            case 68:
            default:
                break;
        }
    }
}

// controller for releasing hotkeys
function onKeyUp(event) {
    if (isRecording) {
        var KeyID = event.keyCode;
        switch (KeyID) {
            default:
                toggleControl(true);
                break;
        }
    }
}

// toggles between move2D and move3D
function toggleControl(b) {

    // controls3.enabled = b;
    // controls3.update();

    controls.enabled = b;
    controls.update();



    controls2.enabled = b;
    controls2.update();

    controls3.enabled= b;
    controls3.update();

    
    camera3.rotation.setFromVector3(camera.rotation.clone());  
    
    if (app.isRedColor == false) {
        app.forceVisualize = true;
        app.bbox_visualization();
        app.forceVisualize = false;
        app.isRedColor = true;
    }

    

}

function clearTable() {
    for (var i = 0; i < boundingBoxes.length; i++) {
        box = boundingBoxes[i];
        deleteRow(box.id);
    }
    id = 0;
}