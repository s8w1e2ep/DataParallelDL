service WeightSync {
    i32 upload(1: string model);
    string download();
    i32 getGlobalStatus();
}
