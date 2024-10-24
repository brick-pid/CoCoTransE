main() {
    try{
        let correct = ArrayList<Bool>(
                (changeBase(8, 3) == "22"),
                (changeBase(9, 3) == "100"),
                (changeBase(234, 2) == "11101010"),
                (changeBase(16, 2) == "10000"),
                (changeBase(8, 2) == "1000"),
                (changeBase(7, 2) == "111")
        );
        if (correct.contains(false)) {
            print(correct)
        }
        for (x in 2..8) {
            if ((changeBase(x, x + 1)!= String(Char(x+Int64(UInt32('0')))))) {
                print("false")
            }
        }
    }
        catch(e: Exception)
    {
        print("false")
    }
}