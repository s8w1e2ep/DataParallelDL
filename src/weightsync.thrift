service WeightSync {
    void upload(1: string model);
    string download();
    i32 get_updateCount();
}
