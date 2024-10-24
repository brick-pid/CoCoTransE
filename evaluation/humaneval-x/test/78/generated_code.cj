main() {
    try{
    let correct = ArrayList<Bool>(
            hexKey("AB") == 1,
            hexKey("1077E") == 2,
            hexKey("ABED1A33") == 4,
            hexKey("2020") == 2,
            hexKey("123456789ABCDEF0") == 6,
            hexKey("112233445566778899AABBCCDDEEFF00") == 12,
            hexKey("") == 0
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