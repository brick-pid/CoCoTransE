main() {
    try{
    let correct = ArrayList<Bool>(
            stringToMd5("Hello world").getOrThrow()=="3e25960a79dbc69b674cd4ec67a72c62",
            stringToMd5("").isNone(),
            stringToMd5("A B C").getOrThrow()=="0ef78513b0cb8cef12743f5aeb35f888",
            stringToMd5("password").getOrThrow()=="5f4dcc3b5aa765d61d8327deb882cf99"
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