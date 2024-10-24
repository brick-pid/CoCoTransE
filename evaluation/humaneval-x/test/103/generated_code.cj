main() {
    try{
    let correct = ArrayList<Bool>(
            roundedAvg(1, 5) == "11",
            roundedAvg(7, 13) == "1010",
            roundedAvg(964, 977) == "1111001011",,
            roundedAvg(996, 997) == "1111100101",,
            roundedAvg(560, 851) == "1011000010",,
            roundedAvg(185, 546) == "101101110",
            roundedAvg(362, 496) == "110101101",
            roundedAvg(350, 902) == "1001110010",,
            roundedAvg(197, 233) == "11010111",
            // unsupported in cangjie
            Int64(roundedAvg(7, 5))  == -1,
            Int64(roundedAvg(5, 1))  == -1,
            roundedAvg(5, 5) == "101"
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