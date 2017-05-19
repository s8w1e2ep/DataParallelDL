service WeightSync {
    i32 upload(1: i32 cnid, 2: string model);
    string download();
    i32 getGlobalStatus();
}
