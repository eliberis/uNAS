machine = edgar@aoraki.cs.ox.ac.uk
project_name = munas
remote_path = "${machine}:/nfs-share/edgar/Projects/${project_name}"

.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf **/__pycache__
	rm -rf **/.pytest_cache

.PHONY: clean-all
clean-all: clean
	rm -rf artifacts
	rm -rf *.out
	rm -rf *.err

.PHONY: upload
upload:
	rsync -avzzu --filter ':- .gitignore' . $(remote_path)

.PHONY: submit-job
submit-job: upload
	ssh $(machine) "cd /nfs-share/edgar/workspace/${project_name} && sbatch slurm_job.sh"

.PHONY: upload-arcus
upload-arcus:
	rsync -avzzu --filter ':- .gitignore' . "arcus-htc:/data/coml-oxmlsys/chri5387/${project_name}"

.PHONY: upload-hpc
upload-hpc:
	rsync -avzzu --filter ':- .gitignore' . "el398@login-gpu.hpc.cam.ac.uk:/home/el398/${project_name}"
