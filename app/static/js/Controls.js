var maxSize = 4;
var pointSize = 2;

var SettingsControls = function() {
    this.Size = pointSize / maxSize;
    this.OutlierRemoval = 'None';
    this.Clustering = 'OriginAwareClustering';
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

    this.SearchRange = 3.5;
    this.NeighborsRadius = 4;

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

var annId = settingsFolder.add(settingsControls, 'AnnotatorId', ['Guest', 'Hasan', 'Akhil','Zuxin', 'Mansur','Guilin','Manoj', 'Runzhong']).onChange(function() {
    // Load Annotation - based on user ID

    if(app && app.cur_frame){
        
        var annotatorID = settingsControls["AnnotatorId"];
        settingsControls["AnnotatorId"] = app.annotatorID;
        app.write_frame_out();
        app.annotatorID = annotatorID;
        settingsControls["AnnotatorId"] = app.annotatorID;
        
        
        $("#annId select").prop("disabled", "disabled");
        app.lock_frame = false;
        var fname = app.cur_frame.fname;
        var framelist = Object.keys(app.frames);

        if(app.cur_frame){
        
            var bboxes = app.cur_frame.bounding_boxes;
            
            for(var j=0; j < bboxes.length; j++){
                
                var box =  bboxes[j];
                deleteRow(box.id);
                box.text_label.element.remove();
                scene.remove(box.points);
                scene.remove(box.boxHelper);

                
                                  
                
                delete bboxes[j];
            }
            app.cur_frame.bounding_boxes = [];
            delete app.cur_frame;
        }
        
        for(var i=0; i < framelist.length; i++){
            var bboxes = app.frames[framelist[i]].bounding_boxes;
            
            for(var j=0; j < bboxes.length; j++){
                
                var box =  bboxes[j];

                box.text_label.element.remove();
                scene.remove(box.points);
                scene.remove(box.boxHelper);

                
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

settingsFolder.add(settingsControls, 'FrameTracking').onChange(function() {

    enable_bounding_box_tracking = settingsControls['FrameTracking'];
});
settingsFolder.add(settingsControls, 'FullyAutomatedBbox').onChange(function() {

    
    
    app.cur_frame.annotated = false;

    if(settingsControls['FullyAutomatedBbox'] == false ){
        
       // deleteAllBoundingBox(false);
    
    }
    
    app.frame_lock = true;
    fname = app.cur_frame["fname"]
    //app.frames = []
    app.fully_automated_bbox(fname);    
    app.frame_lock = false;

    updateCountBBOX();
});




settingsFolder.add(settingsControls, 'FittingCriterion', {
    '(A) Area': 0,
    '(C) Closeness': 1,
    '(V) Variance': 3
});



var PointsFolder = gui.addFolder('Points');

PointsFolder.add(settingsControls, 'Clustering', {
    '(D) DBSCAN': 'DBSCAN',
    '(R) Region Growing': 'OriginAwareClustering'
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

$(".property-name").each(function( index ) {
  var textName = $( this ).text();
$( this ).text( textName.replace(/([A-Z])/g, ' $1').trim() )

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
function onKeyDown2(event) {
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


            case 65: // a key
                settingsControls.FittingCriterion = 0;
                autoDrawModeToggle(true);
                break;
            case 67: // c key
                settingsControls.FittingCriterion = 1;
                autoDrawModeToggle(true);
                break;
            case 68: // d key
                settingsControls.Clustering = "DBSCAN";

                autoDrawModeToggle(false);
                break;

            case 82: // r key
                settingsControls.Clustering = "OriginAwareClustering";

                autoDrawModeToggle(false);
                break;



            case 86: // v key
                settingsControls.FittingCriterion = 3;
                autoDrawModeToggle(true);
                break;


            case 90: // z key
                showPreviousFrameBoundingBoxToggle(true);
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
        switch (KeyID) {


            case 65: // a key
                settingsControls.FittingCriterion = 0;
                autoDrawModeToggle(false);
                break;
            case 67: // c key
                settingsControls.FittingCriterion = 1;
                autoDrawModeToggle(false);
                break;
            case 68: // d key
                //settingsControls.FittingCriterion=2;
                //autoDrawModeToggle(false);
                break;
            case 86: // v key
                settingsControls.FittingCriterion = 3;
                autoDrawModeToggle(false);
                break;


            default:
                toggleControl(true);
                break;
        }
    }

    if (event.ctrlKey) {



        toggleControl(true);

    }
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
    

    return;
    if (b) {

        controls.enabled = b;
        controls.update();



    } else {
        if (move2D) {
            controls.enabled = b;
            controls.update();
        }
    }
}

function clearTable() {
    for (var i = 0; i < boundingBoxes.length; i++) {
        box = boundingBoxes[i];
        deleteRow(box.id);
    }
    id = 0;
}