import voyageai

vo = voyageai.Client()
# This will automatically use the environment variable VOYAGE_API_KEY.
# Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")

result = vo.embed(
    texts=["hello world"], 
    model="voyage-code-3", 
    input_type="document",
    output_dimension=2048,
    output_dtype="float"
)
