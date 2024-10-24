main() {
    try{
    let correct = ArrayList<Bool>(
            getRow(ArrayList<ArrayList<Int64>>(
                    ArrayList<Int64>(1, 2, 3, 4, 5, 6),
                    ArrayList<Int64>(1, 2, 3, 4, 1, 6),
                    ArrayList<Int64>(1, 2, 3, 4, 5, 1)
            ), 1)==ArrayList<ArrayList<Int64>>(ArrayList<Int64>(0, 0), ArrayList<Int64>(1, 4), ArrayList<Int64>(1, 0), ArrayList<Int64>(2, 5), ArrayList<Int64>(2, 0)),
            getRow(ArrayList<ArrayList<Int64>>(
                    ArrayList<Int64>(1, 2, 3, 4, 5, 6),
                    ArrayList<Int64>(1, 2, 3, 4, 5, 6),
                    ArrayList<Int64>(1, 2, 3, 4, 5, 6),
                    ArrayList<Int64>(1, 2, 3, 4, 5, 6),
                    ArrayList<Int64>(1, 2, 3, 4, 5, 6),
                    ArrayList<Int64>(1, 2, 3, 4, 5, 6)
            ), 2)==ArrayList<ArrayList<Int64>>(ArrayList<Int64>(0, 1), ArrayList<Int64>(1, 1), ArrayList<Int64>(2, 1), ArrayList<Int64>(3, 1), ArrayList<Int64>(4, 1), ArrayList<Int64>(5, 1)),
            getRow(ArrayList<ArrayList<Int64>>(
                    ArrayList<Int64>(1, 2, 3, 4, 5, 6),
                    ArrayList<Int64>(1, 2, 3, 4, 5, 6),
                    ArrayList<Int64>(1, 1, 3, 4, 5, 6),
                    ArrayList<Int64>(1, 2, 1, 4, 5, 6),
                    ArrayList<Int64>(1, 2, 3, 1, 5, 6),
                    ArrayList<Int64>(1, 2, 3, 4, 1, 6),
                    ArrayList<Int64>(1, 2, 3, 4, 5, 1)
            ), 1)==ArrayList<ArrayList<Int64>>(ArrayList<Int64>(0, 0), ArrayList<Int64>(1, 0), ArrayList<Int64>(2, 1), ArrayList<Int64>(2, 0), ArrayList<Int64>(3, 2), ArrayList<Int64>(3, 0), ArrayList<Int64>(4, 3), ArrayList<Int64>(4, 0), ArrayList<Int64>(5, 4), ArrayList<Int64>(5, 0), ArrayList<Int64>(6, 5), ArrayList<Int64>(6, 0)),
            getRow(ArrayList<ArrayList<Int64>>(), 1)==ArrayList<ArrayList<Int64>>(),
            getRow(ArrayList<ArrayList<Int64>>(ArrayList<Int64>([1])), 2)==ArrayList<ArrayList<Int64>>(),
            getRow(ArrayList<ArrayList<Int64>>(ArrayList<Int64>(), ArrayList<Int64>([1]), ArrayList<Int64>(1, 2, 3)), 3)==ArrayList<Int64>(ArrayList<Int64>(2, 2))
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