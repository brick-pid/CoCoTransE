main() {
    try{
        let correct = ArrayList<Bool>(
                doubleTheDifference(ArrayList<Int64>()) == 0,
                doubleTheDifference(ArrayList<Int64>(5, 4)) == 25,
                doubleTheDifference(ArrayList<Int64>(0.1, 0.2, 0.3)) == 0,
                doubleTheDifference(ArrayList<Int64>(-10, -20, -30)) == 0,
                doubleTheDifference(ArrayList<Int64>(-1, -2, 8)) == 0,
                doubleTheDifference(ArrayList<Int64>(0.2, 3, 5)) == 34
        );
        if (correct.contains(false)) {
            print(correct)
        }
        var lst = ArrayList<Int64>()
        for (i in -99..=100 where i % 2 == 1 || i % 2 == -1) {
            lst.append(i);
        }
        // int odd_sum = lst.stream().filter(i -> i instanceof Integer p && p % 2 != 0 && p > 0).map(i -> (Integer) i * (Integer) i).reduce(Integer::sum).orElse(0);
        let odd_sum : Int64 = 166650
        assert doubleTheDifference(lst) == odd_sum;
    }
        catch(e: Exception)
    {
        print("false")
    }
}