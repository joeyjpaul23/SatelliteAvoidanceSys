# AEGIS on Oracle Cloud Always Free

This bundle sets up a single Ampere A1 VM running two things:

- **The console:** `python -m aegis.api`, running as a service on `127.0.0.1:8000`. No port other than SSH is open, and you reach the console through an SSH tunnel. It is never exposed publicly, which also sidesteps the question of whether Space-Track's terms allow redistributing its data.
- **An hourly refresh:** `python -m aegis.ingest.spacetrack refresh`, then `python -m aegis.pipeline.crosscheck --archive`. A timer runs it every hour at a random time between :17 and :22; Space-Track asks clients to stay off :00 and :30. The cross-check appends each new public CDM, with AEGIS's prediction beside it, to `/var/lib/aegis/aegis/crosscheck/cdm_public.jsonl`. Over time that file becomes the validation dataset.

Code is copied from your laptop with rsync, not cloned from git, so the VM runs exactly what is on your laptop, including uncommitted work. The env file with your credentials is never part of the copy.

| File | Where it goes on the VM | Purpose |
|---|---|---|
| `push.sh` | runs on the laptop | Syncs `aegis/` and this bundle to `/opt/aegis/app`. If the services are already installed, it also re-runs setup. |
| `setup.sh` | `/opt/aegis/app/deploy/oracle/` | Idempotent. Installs system packages, the `aegis` user, the venv (`pip install -e`), the env file, the units and the helper. |
| `aegis-api.service` | `/etc/systemd/system/` | The console. |
| `aegis-refresh.service`, `.timer` | `/etc/systemd/system/` | The hourly warm-up. It's skipped cleanly until credentials are set. |
| `aegis-spacetrack.sh` | `/usr/local/bin/aegis-spacetrack` | Runs the spacetrack CLI as `aegis`, with the service's environment. |
| `aegis.env.example` | `/etc/aegis/aegis.env` (root:aegis, 0640) | Space-Track credentials. Setup never overwrites an existing copy. |

Caches live under `/var/cache/aegis`, which is `XDG_CACHE_HOME`, so the clients write to `/var/cache/aegis/aegis/{celestrak,socrates,spacetrack}`.

## 1. Create the VM (OCI console)

1. **Choose your home region carefully.** It's fixed when the account is created and can't be changed, and Always Free A1 capacity is only guaranteed there.
2. Go to **Compute → Instances → Create instance**.
3. **Image:** Change image → Canonical Ubuntu **24.04**, the `aarch64` build. For A1 that's often only the **Minimal** image, which is fine. It's the same OS, packages and support window, just without extras. `setup.sh` installs what AEGIS needs, plus an editor and automatic security updates.
4. **Shape:** Change shape → Ampere → **VM.Standard.A1.Flex**. **1 OCPU / 6 GB is enough.** The console screens about 40 objects per request, and the hourly debris download peaks at a few hundred MB. The picker starts at 1 / 6. If it lets you go higher, 2 / 12 gives headroom; if it's capped at 1 / 6, take it. You can resize later: stop the instance, then Edit → shape. The picker should show the *Always Free-eligible* tag.
5. **Networking:** Keep the default VCN and subnet, and leave **Assign a public IPv4 address** on. Don't add any ingress rules. The default security list already allows SSH (22), and Oracle's Ubuntu image firewalls everything else. If you like, you can tighten 22 to your own IP in the security list.
6. **SSH keys:** Paste your public key (`cat ~/.ssh/id_ed25519.pub`), or let Oracle generate a pair and save the private key.
7. The boot volume default (about 47 GB) is inside the free 200 GB. Click **Create**, then copy the public IP once the instance is running.

**"Out of capacity"** is common for A1. You can:

- retry a bit later, since capacity frees up during the day
- switch to another availability domain on the create page, if your region has more than one
- if you asked for more than 1 OCPU / 6 GB, drop back to it

**Or let `launch.sh` create it and keep retrying.** Set up the OCI CLI first: run `uv tool install oci-cli`, then add an API key under My profile → API keys, and save its configuration preview as `~/.oci/config` with `key_file=~/.oci/oci_api_key.pem`. Then run:

```bash
deploy/oracle/launch.sh --plan         # look up subnet, image and ADs; launch nothing
caffeinate -i deploy/oracle/launch.sh  # retry, rotating ADs, until RUNNING; prints the IP
```

It uses the same settings as the console steps above, plus IMDSv2-only and in-transit encryption. It never creates a second `aegis` instance. Capacity is only needed at creation and at start after a *stop*. Reboots keep the host, so never stop the instance from the console.

Optionally, add a host alias to `~/.ssh/config` on the laptop so you don't have to remember the IP:

```
Host aegis-oci
  HostName <public-ip>
  User ubuntu
  IdentityFile ~/.ssh/<your-key>
```

## 2. Install

Run these from the repo root on the laptop:

```bash
deploy/oracle/push.sh <public-ip>                              # or: aegis-oci
ssh ubuntu@<public-ip> sudo bash /opt/aegis/app/deploy/oracle/setup.sh
```

Then on the VM (`ssh ubuntu@<public-ip>`):

```bash
sudoedit /etc/aegis/aegis.env              # SPACETRACK_USER='...'  SPACETRACK_PASS='...'
sudo systemctl restart aegis-api.service   # the console reads the env file only at start
sudo aegis-spacetrack check                # 0 ok, 2 credentials missing, 1 failure
sudo systemctl start aegis-refresh.service # optional: warm the caches now instead of waiting for :17
```

Only type the password into `sudoedit`. Don't paste it into a chat, a commit, or your shell history.

## 3. Open the console

From the laptop:

```bash
ssh -N -L 8000:127.0.0.1:8000 ubuntu@<public-ip>
```

Leave that running and open <http://localhost:8000>. If port 8000 is already in use locally, for example by a local AEGIS, use `-L 8001:127.0.0.1:8000` and open port 8001 instead.

## 4. Upkeep

```bash
systemctl status aegis-api.service
systemctl list-timers aegis-refresh.timer
journalctl -u aegis-api.service -f
journalctl -u aegis-refresh.service --since today
```

After you change code on the laptop, run `deploy/oracle/push.sh <public-ip>` again. Because the services are already installed, it re-runs `setup.sh`, which reinstalls the package and restarts the console. The env file is left alone.

## 5. Idle reclamation

Oracle may reclaim an Always Free instance it considers idle over a 7-day window. The test is low CPU, network, and (on A1) memory use. An hourly warm-up might not be enough activity to avoid that. The dependable fix is to upgrade the account to **Pay As You Go**: Always Free resources stay free, and they're exempt from idle reclamation. Set a budget alert if you do. Don't add CPU-burning jobs to game the metric.
