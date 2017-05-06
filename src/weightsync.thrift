service WeightSync {
    void notifyToStart(1: i32 cnid);
    i32 upload(1: i32 cnid, 2: string model);
    string download();
    i32 getGlobalStatus();
    void getUploadRecord();
}

service WeightForward {
    oneway void forward(1: i32 status, 2: string model);
}
