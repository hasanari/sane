var options = `<select>
    <option value="car">--Car--</option>
    <option value="van">--Van--</option>
    <option value="truck">--Truck--</option>
    <option value="pedestrian">--Pedestrian--</option>
    <option value="cyclist">--Cyclist--</option>
    <option value="sitter">--Sitter--</option>
    <option value="tram">--Tram--</option>
    <option value="misc">--Misc--</option>
    /select`;


// method to add row to object id table
function addFrameRow(fname) {
    $("{0} tbody".format(FRAMES_TABLE)).append(
        "<tr><td><div class='fname'>{0}</div></td></tr>".format(fname)
    );
}
function updateCountBBOX(){
    
    $("#objectIDs").html('<i class="fa fa-list"></i>&nbsp;&nbsp;IDs ('+app.cur_frame.bounding_boxes.length+')');
    
    if(app.cur_frame.bounding_boxes.length>0){
        $("#ClearObjectTable").show();
    }else{
        $("#ClearObjectTable").hide();
    
    }
    
    
    update_point_size();
    
}

// method to add row to object id table
function addObjectRow(box) {
    $("{0} tbody".format(OBJECT_TABLE)).append(
        "<tr><td class='id'><div class='object_row object_row_id'>{0}</td> \
        <td><div class='object_row select_row' id='parent-obj-id-{1}'>{2}</div></td></tr>".format(box.id, box.id, options)
    );
    if (box.object_id) {
        var row = getRow(box.id);
        $(row).find("select").val(box.object_id);
    }
    $("{0} tbody select".format(OBJECT_TABLE)).last().focus();
    updateCountBBOX();


    updateSelectOption(box);
}

function updateSelectOption(box){
    if( box.islocked ){
        $('#parent-obj-id-'+box.id).find("select").attr("disabled", "disabled");
    }else{

        $('#parent-obj-id-'+box.id).find("select").removeAttr("disabled");
    }
}


$(FRAMES_TABLE).on("mousedown", "tbody tr", function() {
    var frameId = $(this).find('.fname').text();
    if (!app.frame_lock) {
        app.tempBBOX = [];
        app.set_frame(frameId);
        $("{0} tbody tr".format(FRAMES_TABLE)).each(
            function(idx, elem) {
                unfocus_frame_row(elem);
            }
        );
        focus_frame_row($(this));
    }
});

function focus_frame_row(frame) {
    $(frame).find(".fname").attr("selected", true);

}

function unfocus_frame_row(frame) {
    $(frame).find(".fname").attr("selected", false);
}


// $(OBJECT_TABLE).on("mousedown", ".object_row_id", function() {
//     $("{0} tbody tr".format(OBJECT_TABLE)).each(
//         function(idx, elem) {
//             unfocus_object_row(elem);
//             // console.log(idx);
//         }
//     );
//     focus_object_row($(this));
// });

function focus_object_row(frame) {
    $(frame).attr("selected", true);
}

function unfocus_object_row(frame) {
    $(frame).attr("selected", false);
    if ($(frame).find("input").length == 1) {
        var boxId = $(frame).find("input").val();
        $(frame).html(boxId);
        console.log($(frame).html());
    }
}

// handler that highlights input and corresponding bounding box when input is selected
$(OBJECT_TABLE).on('mousedown', '.object_row_id', function(e) {
    if (e.target != this || !isRecording) {return false;}
    var is_selected = $(this).attr("selected");
    var is_editing = $(this).find("input").length == 1;
    isMoving = false;
    var boxId = $(this).find("input").length == 1 ? $(this).find("input").val() : $(this).text();
    var box = getBoxById(boxId);
    if (box) {
        $("{0} .object_row_id".format(OBJECT_TABLE)).each(
            function(idx, elem) {
                unfocus_object_row(elem);
            }
        );
        box.select(null);
        
        if (is_selected) {
            app.editing_box_id = true;
        } else {
            selectedBox = box;
            app.editing_box_id = false;
        }
        focus_object_row($(this));
        
        
        app.bbox_visualization();
    }
    });


// handler that saves input when input is changed
$("#object-table").on('change', 'tbody tr', updateObjectId);
//$("#object-table").on('click', 'tbody tr', updateObjectId);
$("#object-table").on('mousedown', 'tbody tr', updateObjectId);
//$("#object-table").on('mouseup', 'tbody tr', updateObjectId);



// method to update Box's object id
function updateObjectId() {
    var boxId, input, box;
    boxId = $(this).find(".object_row_id").text();
    
    box = getBoxById(boxId);
    if (box) {
        input = $(this).find('select').val();

        box.object_id = input;
        box.set_box_id(parseInt(boxId))

        box.add_timestamp();
        
        app.forceVisualize = true;
        selectedBox = box;
        app.bbox_visualization();
        
        app.forceVisualize = false;
        
    }
    
    app.increment_label_count();
    
    updateCountBBOX();
}


// method to get object id table row given id
function getRow(id) {
    var row = $("#object-table tbody").find('td').filter(function() {
        return $(this).text() == id.toString();}).closest("tr");
    return row;
}

// method to select row of object id table given ids
function selectRow(id) {
    var row = getRow(id);    
    $(row).find('select').get(0).focus();
}

function getFrameRow(id) {
    var row = $("{0}".format(FRAMES_TABLE)).find('.fname').filter(function() {
        return $(this).text() == id.toString();}).closest("tr");
    return row;
}

// removes row of object id table given corrensponding bounding box id
function deleteRow(id) {
    var row = getRow(id);
    row.remove();
    
    updateCountBBOX();
    
}

function updateAllObjectIds(){

}


function clearObjectTable() {
    $(OBJECT_TABLE).find('tbody tr').remove();
    updateCountBBOX();
}

function loadObjectTable() {
    if (app.cur_frame) {
        $("#GoToNextFrame").remove();
        for (var i = 0; i < app.cur_frame.bounding_boxes.length; i++) {
            var box = app.cur_frame.bounding_boxes[i];
            addObjectRow(box);
        }
        
        $(OBJECT_TABLE).find('tbody').parent().append('<a href="#" id="GoToNextFrame" alt="Go to next frame!" style="position: absolute;right: 15px;/* bottom: 30px; */margin-top: 13px;" onclick="return gotonextFrame();"><i class="	fa fa-angle-double-right"></i>&nbsp;Next frame</a>')

        $("#GoToNextFrame").focus();
    }
}

// gets box given its id
function getBoxById(id) {
    if (!app.cur_frame) return;
    var boundingBoxes = app.cur_frame.bounding_boxes;
    for (var i = 0; i < boundingBoxes.length; i++) {
        if (boundingBoxes[i].id == id) {
            return boundingBoxes[i];
        }
    }
    return null;
}