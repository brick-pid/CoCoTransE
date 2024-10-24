main() {
    try{
    let correct = ArrayList<Bool>(
            (Int64) compareOne(1, 2).get() == 2,
            (Float64) compareOne(1, 2.5).get() == 2.5,
            (Int64) compareOne(2, 3).get() == 3,
            (Int64) compareOne(5, 6).get() == 6,
            (String) compareOne(1, "2,3").get() == "2,3",
            (String) compareOne("5,1", "6").get() == "6",
            (String) compareOne("1", "2").get() == "2",
            compareOne("1", 1).isEmpty()
    );
if (correct.contains(false)) {
                print(correct)
        }
    }
    catch(e: Exception)
    {
        print("false")
    }
}