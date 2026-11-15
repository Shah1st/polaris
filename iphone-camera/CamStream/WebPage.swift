/// Pages HTML servies par l'iPhone.
enum WebPage {
    /// Page complète (`/`) : le flux + quelques infos et raccourcis.
    static let player = """
    <!doctype html>
    <html lang="fr">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>CamStream</title>
    <style>
    html,body{margin:0;height:100%;background:#000;color:#ddd;font:14px -apple-system,BlinkMacSystemFont,system-ui,sans-serif}
    body{display:flex;flex-direction:column}
    main{flex:1;min-height:0;display:flex;align-items:center;justify-content:center}
    img{max-width:100%;max-height:100%;object-fit:contain}
    footer{display:flex;flex-wrap:wrap;gap:8px 18px;justify-content:center;align-items:center;padding:10px 16px;background:#111;border-top:1px solid #222}
    a,button{color:#7cc4ff;background:none;border:0;padding:0;font:inherit;cursor:pointer;text-decoration:none}
    a:hover,button:hover{text-decoration:underline}
    #dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#e33;margin-right:6px;vertical-align:1px}
    #dot.on{background:#3c3}
    </style>
    </head>
    <body>
    <main><img id="v" src="/stream" alt="Flux caméra de l'iPhone"></main>
    <footer>
    <span><span id="dot"></span><span id="s">Connexion…</span></span>
    <a href="/snapshot.jpg" download="camstream.jpg">Capture</a>
    <a href="/stream" target="_blank">Flux brut</a>
    <a href="/view" target="_blank">Vue seule</a>
    <button id="fs">Plein écran</button>
    </footer>
    <script>
    const v=document.getElementById('v'),s=document.getElementById('s'),dot=document.getElementById('dot');
    let lost=false;
    function reconnect(){v.src='/stream?t='+Date.now();}
    v.onerror=()=>setTimeout(reconnect,1000);
    document.getElementById('fs').onclick=()=>(v.requestFullscreen||v.webkitRequestFullscreen).call(v);
    async function poll(){
      try{
        const r=await fetch('/status',{cache:'no-store'});const j=await r.json();
        s.textContent=`${j.width}×${j.height} · ${j.fps} i/s · caméra ${j.camera==='front'?'avant':'arrière'}${j.lens==='ultrawide'?' grand angle 0,5×':''}${j.depth==='lidar'?' · LiDAR (/depth.jpg)':''}${j.stabilization&&j.stabilization!=='off'?' · stabilisé':''} · ${j.viewers} spectateur${j.viewers>1?'s':''}`;
        dot.className='on';
        if(lost){lost=false;reconnect();}
      }catch(e){s.textContent='iPhone injoignable (app fermée ou en arrière-plan ?)';dot.className='';lost=true;}
    }
    poll();setInterval(poll,2000);
    </script>
    </body>
    </html>
    """

    /// Page minimale (`/view`) : uniquement l'image, idéale pour une « Source navigateur » OBS.
    static let bare = """
    <!doctype html>
    <html lang="fr">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>CamStream</title>
    <style>
    html,body{margin:0;height:100%;background:#000;overflow:hidden}
    img{width:100%;height:100%;object-fit:contain;display:block}
    </style>
    </head>
    <body>
    <img id="v" src="/stream" alt="">
    <script>
    const v=document.getElementById('v');
    v.onerror=()=>setTimeout(()=>{v.src='/stream?t='+Date.now();},1000);
    </script>
    </body>
    </html>
    """
}
