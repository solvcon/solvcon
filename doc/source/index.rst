=======
solvcon
=======

solvcon is a project for solving conservation-law problems by using `the
space-time Conservation Element and Solution Element (CESE) method
<https://yyc.solvcon.net/en/latest/cese/index.html>`__:

.. math::

  \frac{\partial\mathbf{u}}{\partial t}
  + \sum_{k=1}^3 \mathrm{A}^{(k)}(\mathbf{u})
                 \frac{\partial\mathbf{u}}{\partial x_k}
  = 0

where :math:`\mathbf{u}` is the unknown vector and :math:`\mathrm{A}^{(1)}`,
:math:`\mathrm{A}^{(2)}`, and :math:`\mathrm{A}^{(3)}` are the Jacobian
matrices. The code implementation is in https://github.com/solvcon/modmesh.

.. raw:: html

  <script async>
    async function fetchWorkflowData() {
      const baseUrl = 'https://api.github.com/repos/solvcon/modmesh/actions/runs';
      const queryParams = '?event=schedule&status=success';
      const response = await fetch(`${baseUrl}${queryParams}`);
      if (!response.ok) {
        console.error(`Failed to fetch workflow runs: ${response.statusText}`);
        return; // Exit if the initial fetch fails
      }
      const data = await response.json();
      const devbuildRuns = data.workflow_runs.filter(run => run.name === "devbuild");
      if (devbuildRuns.length === 0) {
        console.error('No successful devbuild runs found.');
        return; // Exit if no devbuild runs are found
      }

      let foundArtifact = false; // Flag to track if any valid artifacts are found
      for (const run of devbuildRuns) {
        await new Promise(resolve => setTimeout(resolve, 200)); // delay 200ms between tries to avoid rate limiting
        const artifactsResponse = await fetch(run.artifacts_url);
        if (!artifactsResponse.ok) {
          console.error(`Failed to fetch artifacts for run ID: ${run.id}`);
          continue; // Skip to the next run if the fetch fails
        }
        const artifactsData = await artifactsResponse.json();
        if (artifactsData.artifacts && artifactsData.artifacts.length > 0) {
          const artifactId = artifactsData.artifacts[0].id;
          console.log('Artifact ID:', artifactId);
          const downloadUrl = `https://github.com/solvcon/modmesh/actions/runs/${run.id}/artifacts/${artifactId}`
          console.log('Download URL:', downloadUrl);
          window.location.href = downloadUrl;
          foundArtifact = true;
          break; // Exit the loop after finding the first valid artifact
        } else {
          console.error(`No artifacts found for run ID: ${run.id}`);
        }
      }
      if (!foundArtifact) {
        console.error('No downloadable artifacts were found after checking all eligible devbuild runs.');
      }
    }
  </script>

  <p>
  An experimental pre-built Windows binary for <a
  href="https://github.com/solvcon/modmesh/">modmesh</a> is made in a zip file
  with <a
  href="https://github.com/solvcon/modmesh/actions/workflows/devbuild.yml?query=event%3Aschedule+is%3Asuccess+branch%3Amaster">
  the "devbuild" nightly runs in GitHub Actions</a>. You can scroll down to the
  "artifacts" section in the Windows release run to download it (login to <a
  href="https://github.com/">GitHub</a> is required). Or you can directly click
  <a href="javascript:void(0);" onclick="fetchWorkflowData()">the link that
  points to the nightly build</a> to download.
  </p>

The code remaining in the old repository https://github.com/solvcon/solvcon
will eventually be updated to include setups for problems and solutions. New
contents will be added here after the following are moved to `modmesh
<https://github.com/solvcon/modmesh/>`__ and `its note
<https://github.com/solvcon/mmnote/>`__:

.. toctree::
  :maxdepth: 1

  mesh
  nestedloop
  python_style

.. vim: set spell ft=rst ff=unix fenc=utf8:
