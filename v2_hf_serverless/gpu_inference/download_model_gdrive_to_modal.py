import modal

# 1. Setup the Environment
image = modal.Image.debian_slim().pip_install("gdown")
app = modal.App("google-drive-teleport")
vol = modal.Volume.from_name("awq-volume")

# 2. The Pull Function
@app.function(
    image=image,
    volumes={"/weights": vol},
    timeout=3600  # Give it an hour for 15GB, though it will be much faster
)
def download_from_drive():
    import gdown
    import os

    # REPLACE THIS with your actual Folder ID
    FOLDER_ID = "1PxlliQViWJ8bnhHcG1779K0aB4fq0xmz"
    
    print(f"🛰️ Connecting to Google Drive Folder: {FOLDER_ID}")
    
    # Download the entire folder into the mounted volume
    # --remaining-ok handles cases where partial downloads exist
    gdown.download_folder(
        id=FOLDER_ID, 
        output="/weights/awq-4bit", 
        quiet=False, 
        remaining_ok=True
    )
    
    # CRITICAL: Commit saves the data permanently to the volume
    print("💾 Committing changes to volume...")
    vol.commit()
    print("✅ Transfer Complete! Your weights are now at /weights/awq-4bit")

if __name__ == "__main__":
    modal.run(app)